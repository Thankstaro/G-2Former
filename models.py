import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl.nn.pytorch import SAGEConv
from dgl.utils import expand_as_pair
from dgl.ops import edge_softmax

class GATConv(nn.Module):
    def __init__(
        self,
        node_feats,
        out_feats,
        n_heads=1,
        edge_feats=16,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._n_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm

        # feat fc
        self.src_fc = nn.Linear(self._in_src_feats, out_feats * n_heads, bias=False)
        if residual:
            self.dst_fc = nn.Linear(self._in_src_feats, out_feats * n_heads)
            self.bias = None
        else:
            self.dst_fc = None
            self.bias = nn.Parameter(out_feats * n_heads)

        # attn fc
        self.attn_src_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        if use_attn_dst:
            self.attn_dst_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        else:
            self.attn_dst_fc = None
        if edge_feats > 0:
            self.attn_edge_fc = nn.Linear(edge_feats, n_heads, bias=False)
        else:
            self.attn_edge_fc = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.src_fc.weight, gain=gain)
        if self.dst_fc is not None:
            nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        if self.attn_dst_fc is not None:
            nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
        if self.attn_edge_fc is not None:
            nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat_src, feat_edge=None, lg=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            if self._use_symmetric_norm:
                if not lg:
                    degs = graph.srcdata["deg"]
                else:
                    degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            feat_src_fc = self.src_fc(feat_src).view(-1, self._n_heads, self._out_feats)
            feat_dst_fc = self.dst_fc(feat_dst).view(-1, self._n_heads, self._out_feats)
            attn_src = self.attn_src_fc(feat_src).view(-1, self._n_heads, 1)

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            graph.srcdata.update({"feat_src_fc": feat_src_fc, "attn_src": attn_src})

            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst).view(-1, self._n_heads, 1)
                graph.dstdata.update({"attn_dst": attn_dst})
                graph.apply_edges(fn.u_add_v("attn_src", "attn_dst", "attn_node"))
            else:
                graph.apply_edges(fn.copy_u("attn_src", "attn_node"))

            e = graph.edata["attn_node"]
            if feat_edge is not None:
                attn_edge = self.attn_edge_fc(feat_edge).view(-1, self._n_heads, 1)
                graph.edata.update({"attn_edge": attn_edge})
                e += graph.edata["attn_edge"]
            e = self.leaky_relu(e)

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("feat_src_fc", "a", "m"), fn.sum("m", "feat_src_fc"))

            rst = graph.dstdata["feat_src_fc"]

            if self._use_symmetric_norm:
                if not lg:
                    degs = graph.dstdata["deg"]
                else:
                    degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim())
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.dst_fc is not None:
                rst += feat_dst_fc
            else:
                rst += self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst, inplace=True)

            return rst

def init_proj(in_dim: int, out_dim: int, p: float, norm: str, hid_dim: int=64, final_act: bool=True):
    mlp_list = []

    mlp_list.append(nn.Linear(in_dim, hid_dim, bias=True))
    if norm == 'bn':
        mlp_list.append(nn.BatchNorm1d(hid_dim))
    elif norm == 'ln':
        mlp_list.append(nn.LayerNorm(hid_dim))
    mlp_list.append(nn.ReLU())
    mlp_list.append(nn.Dropout(p=p))
    mlp_list.append(nn.Linear(hid_dim, out_dim, bias=True))
    mlp_list.append(nn.Dropout(p=p))
    if final_act:
        if norm == 'bn':
            mlp_list.append(nn.BatchNorm1d(hid_dim))
        elif norm == 'ln':
            mlp_list.append(nn.LayerNorm(hid_dim))
        mlp_list.append(nn.Sigmoid())

    return nn.Sequential(*mlp_list)


class MultiFConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm,
                 dropout,
                 act=F.sigmoid):
        super(MultiFConv, self).__init__()
        self.act = act
        # self.linear = nn.Linear(in_feats, out_feats)
        self.dropout = nn.Dropout(p=dropout)
        # self.qlinear = nn.Linear(in_feats, out_feats)
        # if norm == 'bn':
        #     self.norm = nn.BatchNorm1d(out_feats)
        # elif norm == 'ln':
        #     self.norm = nn.LayerNorm(out_feats)

    def Conv(self, q, feat, high):
        D_invsqrt = torch.pow(torch.mm(q, torch.mm(q.t(), torch.ones(feat.shape[0], 1, device=feat.device)))-1+1e-10, -0.5)
        
        h = torch.mm(q.t(), feat*D_invsqrt)
        h = torch.mm(q, h)
        h = D_invsqrt * h
        h = h - (D_invsqrt * D_invsqrt * feat)

        if high:
            return feat - h
        else:
            return h + feat

    def forward(self, q, feat, theta):
        
        h = feat
        for i in range(theta[1]):
            h = (i + 1) * 0.5/(i + 1) * self.Conv(q, h, high=False)
        for j in range(theta[0]):
            h = (theta[1] + j + 1) * 0.5/(j + 1) * self.Conv(q, h, high=True)
        h = (sum(theta) + 1) * 0.5 * h
        # h_logit = self.linear(h)
        # h_logit = self.dropout(h_logit)
        
        return h


class CosGNN(nn.Module):
    def __init__(self, dataset, L, K, input_size, hidden_size, num_o, indropout, dropout, GNN,
                 dot = False,
                 norm='bn', 
                 res=False, 
                 mlp_in=False, 
                 mlp_out=True,
                 mode = 'full'):
        super().__init__()

        self.dataset = dataset
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.L = L
        self.K = K
        self.mode = mode
        self.num_o = num_o
        self.act = F.relu
        self.dropout = nn.Dropout(p=dropout)
        self.indropout = nn.Dropout(p=indropout)
        self.norm = norm
        self.res = res
        self.dot = dot
        self.mlp_out = mlp_out
        self.mlp_in = mlp_in
        self.GNN = GNN
        self.thetas = self.get_thetas(K)
        self.channs = nn.ModuleList()
        if GNN == 'gcn':
            self.linears = nn.ModuleList()
        elif GNN =='sage':
            self.sageconvs = nn.ModuleList()
        else:
            self.gatconvs = nn.ModuleList()
        if res:
            self.reslins = nn.ModuleList()
        if norm is not None:
            self.norms = nn.ModuleList()
        # self.prelns = nn.ModuleList()
        if norm == 'bn':
            self.norms.append(nn.BatchNorm1d(hidden_size))
        elif norm == 'ln':
            self.norms.append(nn.LayerNorm(hidden_size))
        if mlp_in:
            self.proj_in = nn.Linear(input_size, hidden_size)
            self.proj = init_proj(hidden_size, hidden_size, dropout, norm, hidden_size)
            self.proj2 = init_proj(hidden_size, hidden_size, dropout, norm, hidden_size, False)
            self.proj_chan = nn.Linear(hidden_size, num_o)
            if GNN == 'gcn':
                self.linears.append(nn.Linear(hidden_size+num_o*0, hidden_size, bias=False))
            elif GNN == 'sage':
                self.sageconvs.append(SAGEConv(hidden_size+num_o*0, hidden_size, 'mean'))
            else:
                self.gatconvs.append(GATConv(hidden_size+num_o*0, int(hidden_size/2), n_heads=2))
            # self.linears2.append(nn.Linear(hidden_size, hidden_size, bias=False))
            if res:
                self.reslins.append(nn.Linear(hidden_size+num_o*0, hidden_size))
        else:
            if GNN == 'gcn':
                self.linears.append(nn.Linear(input_size+num_o*0, hidden_size, bias=False))
            elif GNN == 'sage':
                self.sageconvs.append(SAGEConv(input_size+num_o*0, hidden_size, 'mean'))
            else:
                self.gatconvs.append(GATConv(input_size+num_o*0, int(hidden_size/2), n_heads=2))
            self.proj = init_proj(input_size, hidden_size, dropout, norm, hidden_size)
            self.proj2 = init_proj(input_size, input_size, dropout, norm, hidden_size, False)
            self.proj_chan = nn.Linear(input_size, num_o)
            if res:
                self.reslins.append(nn.Linear(input_size+num_o*0, hidden_size))
        for i in range(L-1):
            if GNN == 'gcn':
                self.linears.append(nn.Linear(hidden_size, hidden_size, bias=False))
            elif GNN == 'sage':
                self.sageconvs.append(SAGEConv(hidden_size+num_o*0, hidden_size, 'mean'))
            else:
                self.gatconvs.append(GATConv(hidden_size, int(hidden_size/2), n_heads=2))
            # self.linears2.append(nn.Linear(hidden_size, hidden_size, bias=False))
            if res:
                self.reslins.append(nn.Linear(hidden_size, hidden_size))
            if norm == 'bn':
                self.norms.append(nn.BatchNorm1d(hidden_size))
            elif norm == 'ln':
                self.norms.append(nn.LayerNorm(hidden_size))
        for i in range(K+1):
            if mlp_in:
                self.channs.append(MultiFConv(hidden_size, hidden_size, dropout=dropout, norm=norm))
            else:
                self.channs.append(MultiFConv(input_size, hidden_size, dropout=dropout, norm=norm))
        if mlp_out:
            self.proj_out = nn.Linear(hidden_size, num_o)
    
    def get_thetas(self, K):
        thetas = []
        for i in range(K+1):
            if self.mode == 'full':
                thetas.append([i, K-i])
            else:
                thetas.append([i, K])
        
        return thetas

    def wconv(self, g, feat, lg=False):
        with g.local_scope():
            if not lg:
                D_invsqrt = torch.pow(g.srcdata['deg'], -0.5).unsqueeze(-1)
                g.srcdata['h'] = feat[:g.num_src_nodes()].clone() * D_invsqrt
                # g.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))

                return g.dstdata.pop('h') * torch.pow(g.dstdata['deg'], -0.5).unsqueeze(-1)
            else:
                D_invsqrt = torch.pow(g.in_degrees().float(), -0.5).unsqueeze(-1)
                g.ndata['h'] = feat.clone() * D_invsqrt
                # g.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))

                return g.ndata.pop('h') * torch.pow(g.in_degrees().float(), -0.5).unsqueeze(-1)
            
 
    def forward(self, g, lg=False, val=False):
        if not lg:
            final_num = g[-1].num_dst_nodes()
            h = g[0].srcdata['feat']
        else:
            h = g.ndata['feat']
            final_num = h.shape[0]
        h = self.indropout(h)
        if self.mlp_in:
            h = self.proj_in(h)
            # h = self.act(h)
            h = self.indropout(h)
        
        q = self.proj(h)
        q = torch.pow(torch.norm(q, dim=1, keepdim=True)+1e-10, -1) * q
        # D = torch.mm(q, torch.mm(q.t(), torch.ones(q.shape[0], 1, device=q.device)))
        
        channs_hs = 0.
        hs = self.proj2(h)
        # channs_hs.append(hs)
        for i in range(len(self.thetas)):
            ht = self.channs[i](q, hs, self.thetas[i])
            # channs_hs.append(ht)
            channs_hs = channs_hs + ht
        # h = torch.cat([h] + channs_hs, 1)
        if self.dot:
            confin = F.softmax(self.proj_chan(channs_hs*h), 1)
            confin = confin.max(1, keepdim=True).values
            h = ((1.-confin) * h) + (confin*(channs_hs*h))
        else:
            confin = F.softmax(self.proj_chan(channs_hs), 1)
            confin = confin.max(1, keepdim=True).values
            h = ((1.-confin) * h) + (confin*(channs_hs))

        for i in range(self.L):
            if self.GNN == 'gcn':
                if not lg:
                    x = self.wconv(g[i], h)
                else:
                    x = self.wconv(g, h, lg=lg)
                x = self.linears[i](x)
            elif self.GNN == 'sage':
                if not lg:
                    x = self.sageconvs[i](g[i], h)
                else:
                    x = self.sageconvs[i](g, h)
            else:
                if not lg:
                    x = self.gatconvs[i](g[i], h).flatten(1, -1)
                else:
                    x = self.gatconvs[i](g, h, lg=lg).flatten(1, -1)
            
            if self.res:
                if not lg:
                    h = x + self.reslins[i](h[:g[i].num_dst_nodes()])
                else:
                    h = x + self.reslins[i](h)
            else:
                h = x
            if self.norm is not None:
                h = self.norms[i](h)
            h = self.act(h)
            h = self.dropout(h)
            # h[:blocks[i].num_dst_nodes()] = hs.clone()

        pred = self.proj_out(h)
        
        if not val:
            return pred, [self.proj_chan(channs_hs)]
        else:
            return pred[:final_num]

