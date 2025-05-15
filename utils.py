from models import CosGNN, Test2, Test
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import logging
import torch.nn.functional as F
import time
import random
import os
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, accuracy_score
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import scipy.sparse as sp
import copy
from sklearn.metrics import f1_score
import seaborn as sns
import pandas as pd
import dgl

logging.getLogger().setLevel(logging.INFO)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def cal_best_f1(labels, logits):
    # best_f1, best_thre = 0, 0
    # for thre in np.linspace(0.05, 0.95, 19):
    #     preds = np.zeros_like(labels)
    #     preds[logits >= thre] = 1
    #     tmp = f1_score(labels, preds, average='macro')
    #     if tmp > best_f1:
    #         best_f1 = tmp
    #         best_thre = thre
    precision, recall, thresholds = precision_recall_curve(labels, logits)
    F1 = 2 * precision * recall / (precision + recall + 1e-18)
    idx = F1.argmax(axis=0)
    best_thre = thresholds[idx]
    preds = np.zeros_like(labels)
    preds[logits > best_thre] = 1
    best_f1 = f1_score(labels, preds, average='macro')

    return best_f1, best_thre


def cal_lp(A_T):
    D = np.array(A_T.sum(1, dtype=float))
    D_sqrt = np.power(D, -0.5)
    D_sqrt = sp.diags(D_sqrt.flatten())
    A = D_sqrt @ A_T @ D_sqrt

    return A.tocoo()
        

def drop(Adj, mask, labels):
    from scipy.sparse import coo_matrix
    row, col = Adj.coalesce().indices()[0].numpy(), Adj.coalesce().indices()[1].numpy()
    mask1, mask2 = mask.numpy()[row], mask.numpy()[col]
    maskf1 = np.logical_and(mask1, mask2)
    tmp = labels[row] + labels[col]
    maskf2 = tmp == 1
    values = np.ones_like(row)
    maskf = np.logical_and(maskf1, maskf2)
    A = coo_matrix((values, (row, col)), shape=(Adj.shape[0], Adj.shape[0]))
    _, A = cal_lp(A)
    Adj = torch.sparse_coo_tensor(torch.LongTensor(np.array([A.row[~maskf], A.col[~maskf]])), 
                                  torch.FloatTensor(values[~maskf]), Adj.size())
    return Adj


    

def cal_gopt(A, a=1.0):
    A_T = copy.deepcopy(A.transpose())
    A_T.setdiag(0)
    A_T = (A_T + A) / 2
    # A1 = sp.triu(A_T)
    # A2 = sp.tril(A_T)
    # D = np.array(A_T.sum(1, dtype=int)).flatten()
    # row1, col1, row2, col2 = A1.tocoo().row, A1.tocoo().col, A2.tocoo().row, A2.tocoo().col
    # from scipy.sparse import coo_matrix
    # mask1 = (D[row1] - D[col1]) <= 0
    # mask2 = (D[row2] - D[col2]) < 0
    # nrow, ncol = np.concatenate((row1[mask1], row2[mask2])), np.concatenate((col1[mask1], col2[mask2]))
    # A_u = coo_matrix((np.ones_like(nrow), (nrow, ncol)), shape=(A.shape[0], A.shape[0]))
    # mask1 = (D[row1] - D[col1]) > 0
    # mask2 = (D[row2] - D[col2]) >= 0
    # nrow, ncol = np.concatenate((row1[mask1], row2[mask2])), np.concatenate((col1[mask1], col2[mask2]))
    # A_l = coo_matrix((np.ones_like(nrow), (nrow, ncol)), shape=(A.shape[0], A.shape[0]))

    # A_u += sp.identity(A_u.shape[0])
    # A_l += sp.identity(A_l.shape[0])

    A_T += sp.identity(A_T.shape[0])

    A_u = sp.triu(A_T)
    A_l = sp.tril(A_T)

    L_u, Adj_u = cal_lp(A_u)
    # L_u = (L_u + sp.identity(L_u.shape[0]))
    L_l, Adj_l = cal_lp(A_l)
    # L_l = (L_l + sp.identity(L_l.shape[0]))

    # L_mer = (L_l + L_u).tocoo()
    L, _ = cal_lp(A_T)
    # L = (L + sp.identity(L.shape[0]))

    I = sp.identity(L.shape[0]).tocoo()
    # temp = Adj_l.tocsr() @ Adj_u.tocsr()
    # Adj_u = sp.coo_matrix(Adj.toarray()*temp.toarray())
    Adj = (I * (1 + a) - L).tocoo()
    # Adj_u = (I*(1+1)-L_u).tocoo()
    # Adj_l = (I * (1 + 1) - L_l).tocoo()

    L_u = torch.sparse_coo_tensor(torch.LongTensor([L_u.row.tolist(), L_u.col.tolist()]),
                                   torch.FloatTensor(L_u.data.astype(np.float64)),
                                   torch.Size([L_u.shape[0], L_u.shape[0]]))
    L_l = torch.sparse_coo_tensor(torch.LongTensor([L_l.row.tolist(), L_l.col.tolist()]),
                                   torch.FloatTensor(L_l.data.astype(np.float64)),
                                   torch.Size([L_l.shape[0], L_l.shape[0]]))
    L = torch.sparse_coo_tensor(torch.LongTensor([L.row.tolist(), L.col.tolist()]),
                                 torch.FloatTensor(L.data.astype(np.float64)), torch.Size([L.shape[0], L.shape[0]]))
    Adj_u = torch.sparse_coo_tensor(torch.LongTensor([Adj_u.row.tolist(), Adj_u.col.tolist()]),
                                     torch.FloatTensor(Adj_u.data.astype(np.float64)),
                                     torch.Size([Adj_u.shape[0], Adj_u.shape[0]]))
    Adj = torch.sparse_coo_tensor(torch.LongTensor([Adj.row.tolist(), Adj.col.tolist()]),
                                   torch.FloatTensor(Adj.data.astype(np.float64)),
                                   torch.Size([Adj.shape[0], Adj.shape[0]]))
    Adj_l = torch.sparse_coo_tensor(torch.LongTensor([Adj_l.row.tolist(), Adj_l.col.tolist()]),
                                     torch.FloatTensor(Adj_l.data.astype(np.float64)),
                                     torch.Size([Adj_l.shape[0], Adj_l.shape[0]]))

    return [L_u, L_l, L, Adj_u, Adj_l, Adj]


def cal_nA(A):
    A = (A + sp.identity(A.shape[0])).transpose()
    D = np.array(A.sum(1, dtype=float))
    D_sqrt = np.power(D, -1)
    D_sqrt = sp.diags(D_sqrt.flatten())
    A = D_sqrt @ A
    A = A.tocoo()
    A = torch.sparse.FloatTensor(torch.LongTensor([A.row.tolist(), A.col.tolist()]),
                                 torch.FloatTensor(A.data.astype(np.float64)))
    return A


def vis(X, labels):
    tsne_emb = TSNE(n_components=2).fit_transform(X)
    col = []
    for i in labels:
        if i == 0:
            col.append('normal')
        else:
            col.append('abnormal')
    # for i in test_mask:
    #     col[i] = 'g'
    # plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=col, s=10, alpha=0.5)
    data = pd.DataFrame({
        'x': tsne_emb[:, 0],
        'y': tsne_emb[:, 1],
        'labels': col
    })
    ax = sns.scatterplot(x='x', y='y', hue='labels', data=data)
    ax.set_axis_off()
    plt.savefig('F:/K-GAE/image/ama_U_drop.PNG')
    plt.show()


def unsupervised_loss(input, pos_idx, neg_idx):
    pos_loss = F.binary_cross_entropy(input[pos_idx], torch.ones_like(input[pos_idx]))
    neg_loss = F.binary_cross_entropy(input[neg_idx], torch.zeros_like(input[neg_idx]))
    return pos_loss + neg_loss

def hetero(Adj, A, labels, dataset):
    hete = 0.0
    homo = 0.0
    num1, num2 = 0, 0
    row, col = Adj.row.tolist(), Adj.col.tolist()
    if dataset == 'Amazon':
        for i in range(len(row)):
            if labels[row[i]] != labels[col[i]] and row[i] >= 3305 and col[i] >= 3305:
                hete += A[row[i], col[i]]
                num1 += 1
            if labels[row[i]] == labels[col[i]] and row[i] >= 3305 and col[i] >= 3305:
                homo += A[row[i], col[i]]
                num2 += 1
    else:
        for i in range(len(row)):
            if labels[row[i]] != labels[col[i]]:
                hete += A[row[i], col[i]]
                num1 += 1
            else:
                homo += A[row[i], col[i]]
                num2 += 1
    hete = hete/num1
    homo = homo/num2
    print(hete)
    print(homo)
                
    return hete/(hete+homo)

def hetero2(Adj, x, labels, dataset):
    hete = 0.0
    homo = 0.0
    row, col = Adj.row, Adj.col
    # dis = 1 - F.cosine_similarity(x[row], x[col])
    if dataset=='Amazon':
        mask = np.logical_and(row>=3305, col>=3305)
        ind = labels[row[mask]] + labels[col[mask]]
        dis = F.pairwise_distance(x[row[mask]], x[col[mask]])
    else:
        dis = F.pairwise_distance(x[row], x[col])
        ind = labels[row] + labels[col]
    hete = (dis[ind==1]).mean()
    homo = (dis[ind!=1]).mean()
    homo1 = (dis[ind==0]).mean()
    homo2 = (dis[ind==2]).mean()
    print(hete)
    print(homo)
    print([homo1, homo2])
                
    return hete/(hete+homo)

def adjust_order(graph, ranidx):
    g = copy.deepcopy(graph)
    g.ndata['feature'][ranidx] = g.ndata['feature'].clone()
    for r in range(len(g.canonical_etypes)):
        src, dst = g.edges(etype=g.etypes[r])
        nsrc, ndst = ranidx[src], ranidx[dst]
        edges = g.edges(etype=g.etypes[r])
        g.remove_edges(g.edge_ids(edges[0], edges[1], etype=g.etypes[r]), etype=g.etypes[r])
        g.add_edges(nsrc, ndst, etype=g.etypes[r])
    
    edge_dict = {}
    weight = []
    for r in range(len(g.canonical_etypes)):
        gsp = g[g.canonical_etypes[r]].adj_external(scipy_fmt='coo')
        gsp += sp.identity(gsp.shape[0])
        A_u = cal_lp(sp.triu(gsp))
        A_l = cal_lp(sp.tril(gsp))
        weight.append(A_l.data)
        weight.append(A_u.data)
        edge_dict[(g.canonical_etypes[r][0], g.canonical_etypes[r][1]+'_u', g.canonical_etypes[r][2])] = (A_u.row, A_u.col)
        edge_dict[(g.canonical_etypes[r][0], g.canonical_etypes[r][1]+'_l', g.canonical_etypes[r][2])] = (A_l.row, A_l.col)

    g_lu = dgl.heterograph(edge_dict)
    g_lu.ndata['index'] = graph.ndata['index']
    g_lu.ndata['feature'] = g.ndata['feature']
    g_lu.ndata['label'] = g.ndata['label']
    g_lu.ndata['label'][ranidx] = g_lu.ndata['label'].clone()

    for e in range(len(g_lu.etypes)):
        g_lu.edges[g_lu.etypes[e]].data['weight'] = torch.tensor(weight[e], dtype=torch.float32)
    
    g_lu.ndata['train_mask'] = graph.ndata['train_mask'].clone()
    g_lu.ndata['train_mask'][ranidx] = g_lu.ndata['train_mask'].clone()

    return g_lu

def mixup(g, x, labels):
    srcs, dsts = [], []
    for e in range(0, len(g.etypes), 2):
        src, dst = g.edges(etype=g.etypes[e])
        srcs.append(src)
        dsts.append(dst)
    srcs, dsts = torch.cat(srcs), torch.cat(dsts)
    dis = (F.pairwise_distance(x[srcs], x[dsts])).unsqueeze(1)
    dis = torch.cat(((1+dis).pow(-1), 1 - (1+dis).pow(-1)), 1).log()
    soft_targets = labels[srcs] + labels[dsts]
    soft_targets[soft_targets!=1] = 0
    # soft_targets = torch.cat((1-soft_targets.unsqueeze(1), soft_targets.unsqueeze(1)), 1)
    return dis, soft_targets


def train(ds, args, G_opt=None, unsup=False):
    labels, fea = ds.graph.ndata['label'], ds.graph.ndata['feat']

    device = args.device
    gnn = CosGNN(args.dataset, args.L, args.K, np.size(fea, 1), args.hidden_size, args.num_o, args.indropout, args.dropout, args.GNN, 
                 args.dot, 
                 args.norm, 
                 args.res, 
                 args.mlp_in,
                 mode=args.mode)
    # gnn = Test(args.n_gnnlayer, np.size(fea, 1), args.hidden_size, G_opt, args.num_o, unsup, args.device, args.gamma, random_indices)
    # gnn = Test2(np.size(fea, 1), args.hidden_size, G_opt, args.num_o, 'gcn', args.device)
    gnn = gnn.to(device)

    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    train_sampler = dgl.dataloading.NeighborSampler([-1 for _ in range(args.L+args.Delta)])
    val_sampler = dgl.dataloading.NeighborSampler([-1 for _ in range(args.L+args.Delta)])
    test_sampler = dgl.dataloading.NeighborSampler([-1 for _ in range(args.L+args.Delta)])
    best_metric, final_metric = 0, 0
    patience = 0
    t = time.time()
    times = []

    dataloader = dgl.dataloading.DataLoader(ds.graph,
                                            torch.arange(labels.shape[0])[ds.graph.ndata['train_mask']],
                                            train_sampler,
                                            device=device,
                                            batch_size=args.batch_size, 
                                            shuffle=True, 
                                            drop_last=False,
                                            use_uva=True,
                                            num_workers=0)

    dataloader2 = dgl.dataloading.DataLoader(ds.graph,
                                            torch.arange(labels.shape[0])[ds.graph.ndata['val_mask']],
                                            val_sampler,
                                            device=device,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            use_uva=True,
                                            num_workers=0)

    dataloader3 = dgl.dataloading.DataLoader(ds.graph,
                                             torch.arange(labels.shape[0])[ds.graph.ndata['test_mask']],
                                             test_sampler,
                                             device=device,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             use_uva=True,
                                             num_workers=0)

    if args.dataset == 'ogbn-proteins':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.NLLLoss()
    
    for epoch in range(args.epochs):
        logging.info('-------------epoch {}/{}-------------'.format(epoch + 1, args.epochs))
        
        # t = time.time()
        gnn.train()

        train_loss = 0.
        # g_lu = g_lu.to(device)
        for step, (_, output_nodes, blocks) in enumerate(dataloader):
            # unlabel_idx = unlabel_idx.to(device)
            # _, _, blocks = sampler.sample_blocks(g_lu, g_lu.ndata['index'][unlabel_idx])

            optimizer.zero_grad()
            output, chans = gnn(blocks)

            if args.dataset == 'ogbn-proteins':
                loss = criterion(output[:blocks[-1].num_dst_nodes()], blocks[-1].dstdata['label'])
            else:
                pl = torch.argmax(F.softmax(output, 1), 1)
                loss = criterion(F.log_softmax(output[:blocks[-1].num_dst_nodes()], 1), blocks[-1].dstdata['label'])
                for chan in chans:
                    loss = loss + args.gamma*criterion(F.log_softmax(chan[:blocks[-1].num_dst_nodes()], 1), blocks[-1].dstdata['label'])
            
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_loss /= (step+1)

        logging.info('-------------epoch {} is completed.'.format(epoch + 1))
        if (epoch + 1)%100 == 0:
            times.append(time.time() - t)
        
        pred, val_labels = model_val(gnn, dataloader2, args)
        if args.metric == 'Acc':
            stop_metric = accuracy_score(val_labels.numpy(), pred.numpy().argmax(1))
        else:
            stop_metric = roc_auc_score(val_labels.numpy(), pred.numpy())

        torch.cuda.empty_cache()

        if best_metric < stop_metric:
            best_metric = stop_metric
            final_model = copy.deepcopy(gnn)
            patience = 0
            bepoch = epoch + 1
        else:
            patience += 1


        logging.info(('------------- train loss: {:.4f}, vbmetric: {:.4f}, bepoch: {}, time: {:.4f}s'.format(train_loss,
                                                                                                       best_metric,
                                                                                                       bepoch,
                                                                                                       time.time() - t)))

        if args.patience == patience:
            break

        pred, test_labels = model_val(final_model, dataloader3, args)
        
        if args.metric == 'Acc':
            final_metric = accuracy_score(test_labels.numpy(), pred.numpy().argmax(1))
        else:
            final_metric = roc_auc_score(test_labels.numpy(), pred.numpy())
        logging.info(('------------- f_metric: {:.4f}'.format(final_metric)))
    
    return gnn, final_metric

@torch.no_grad
def model_val(model, dataloader, args):
    model.eval()
    
    targets, preds = [], []
    for step, (_, output_nodes, blocks) in enumerate(dataloader):
        # _, _, blocks = sampler.sample_blocks(graph, graph.ndata['index'][node_idx])
        
        pred, q = model(blocks, val=True)
        target = blocks[-1].dstdata['label']
        targets.append(target.detach())
        preds.append(pred.detach())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    if args.dataset == 'ogbn-proteins':
        preds = torch.sigmoid(preds)
    else:
        preds = torch.softmax(preds, 1)
        if args.metric == 'AUC':
            preds = preds[:, 1]

    return preds.cpu(), targets.cpu()

def train_lg(ds, args, G_opt=None, unsup=False):
    labels, fea = ds.graph.ndata['label'], ds.graph.ndata['feat']

    device = args.device
    gnn = CosGNN(args.dataset, args.L, args.K, np.size(fea, 1), args.hidden_size, args.num_o, args.indropout, args.dropout, args.GNN, 
                 args.dot, 
                 args.norm, 
                 args.res, 
                 args.mlp_in,
                 mode=args.mode)
    gnn = gnn.to(device)

    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_metric, final_metric = 0, 0
    patience = 0
    t = time.time()
    times = []

    criterion = torch.nn.NLLLoss()

    num_batch = labels.shape[0] // args.batch_size + 1

    train_losses, test_losses = [], []
    
    for epoch in range(args.epochs):
        logging.info('-------------epoch {}/{}-------------'.format(epoch + 1, args.epochs))
        
        # t = time.time()
        gnn.train()

        train_loss = 0.
        idx = torch.randperm(labels.shape[0])
        for i in range(num_batch):
            idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
            optimizer.zero_grad()
            subg = dgl.node_subgraph(ds.graph, idx_i, output_device=device)
            output, chans = gnn(subg, lg=True)

            loss = criterion(F.log_softmax(output[subg.ndata['train_mask']], 1), subg.ndata['label'][subg.ndata['train_mask']])
            for chan in chans:
                loss = loss + args.gamma*criterion(F.log_softmax(chan[subg.ndata['train_mask']], 1), subg.ndata['label'][subg.ndata['train_mask']])
            
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_loss /= num_batch
        train_losses.append(train_loss)


        logging.info('-------------epoch {} is completed.'.format(epoch + 1))
        if (epoch + 1)%100 == 0:
            times.append(time.time() - t)
        
        val_metric, test_metric, pred, test_labels = model_val_lg(gnn, args, ds.graph)
        # with torch.no_grad():
        #     test_losses.append(criterion(F.log_softmax(pred, 1), test_labels).item())

        torch.cuda.empty_cache()

        if best_metric < val_metric:
            best_metric = val_metric
            final_metric = test_metric
            patience = 0
            bepoch = epoch + 1
        else:
            patience += 1


        logging.info(('------------- train loss: {:.4f}, vbmetric: {:.4f}, bepoch: {}, time: {:.4f}s'.format(train_loss,
                                                                                                       best_metric,
                                                                                                       bepoch,
                                                                                                       time.time() - t)))

        if args.patience == patience:
            break

        logging.info(('------------- f_metric: {:.4f}'.format(final_metric)))
    # print(train_losses)
    # print(test_losses)
    return gnn, final_metric

@torch.no_grad
def model_val_lg(model, args, graph):
    model.eval()
    graph = graph.to(args.device)
    targets, preds = [], []
    preds = model(graph, lg=True, val=True)
    targets = graph.ndata['label']
    predsoh = preds.clone()

    if args.metric == 'AUC':
        preds = preds[:, 1]
        preds = torch.sigmoid(preds)
    else:
        preds = torch.softmax(preds, 1).argmax(1)
    val_preds, test_preds = preds[graph.ndata['val_mask']].cpu(), preds[graph.ndata['test_mask']].cpu()
    val_labels, test_labels = targets[graph.ndata['val_mask']].cpu(), targets[graph.ndata['test_mask']].cpu()
    
    if args.metric == 'Acc':
        val_metric = accuracy_score(val_labels.numpy(), val_preds.numpy())
        test_metric = accuracy_score(test_labels.numpy(), test_preds.numpy())
    else:
        val_metric = roc_auc_score(val_labels.numpy(), val_preds.numpy())
        test_metric = roc_auc_score(test_labels.numpy(), test_preds.numpy())
        
    return val_metric, test_metric, predsoh[graph.ndata['test_mask']], targets[graph.ndata['test_mask']]