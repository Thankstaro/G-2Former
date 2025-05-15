import dgl.data
import scipy as sp
import numpy as np
import pickle
import torch
import scipy.io as scio
from dgl.data.utils import load_graphs
from dgl.data.utils import save_graphs
from torch.utils.data import DataLoader as torch_dataloader
from sklearn.model_selection import train_test_split
import pandas as pd
import dgl
from os import path
from utils import cal_lp
import networkx as nx
from dgl import RowFeatNormalizer
from ogb.nodeproppred import DglNodePropPredDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, WikiCS
import torch_geometric.transforms as T

class DataLoader(object):
    def __init__(self, args, i=None):
        self.file_paths = f'{args.file_paths}'
        self.name = args.dataset
        self.batch_size = args.batch_size
        self.indices = None
        self.graph = self.load_data(args, i)
        if self.indices is not None:
            self.graph = self.split(self.graph, self.indices)
        
        # from sklearn.cluster import kmeans_plusplus
        # from sklearn.neighbors import NearestNeighbors
        # import torch.nn.functional as F

        # X = self.graph.ndata['feat'].clone()
        # X = torch.pow(torch.norm(X, dim=1, keepdim=True)+1e-10, -1) * X
        # centroid, _ = kmeans_plusplus(X.numpy(), n_clusters=200)
        # centroid = torch.from_numpy(centroid).float()
        # centroid = torch.pow(torch.norm(centroid, dim=1, keepdim=True)+1e-10, -1) * centroid

        # k = 50
        # model = NearestNeighbors(n_neighbors=k, metric='cosine')
        # model.fit(centroid)
        # _, indices = model.kneighbors(X)
        # S = F.relu(torch.mm(X, centroid.t()))
        # mask = torch.zeros_like(S)
        # row_idx = torch.arange(X.shape[0]).unsqueeze(1).expand(-1, k)
        # mask[row_idx, indices] = 1.
        # S = S * mask
        # S = S / (S.sum(1, keepdim=True)+1e-10)
        # L, Q = torch.linalg.eigh(torch.mm(S.t(), S))
        # L = torch.clamp(L, min=0.0)
        # U = torch.mm(S, Q) / torch.sqrt(L).unsqueeze(0)
        # PE = U[:, 0:args.num_o].contiguous()
        # self.graph.ndata['PE'] = PE


    def normlize(self, ds):
        m = ds.mean(0)
        std = ds.std(0)
        return (ds-m)/(std+1e-10)
    
    def class_rand_splits(self, label, label_num_per_class, valid_num=500, test_num=1000):
        """use all remaining data points as test data, so test_num will not be used"""
        train_idx, non_train_idx = [], []
        idx = torch.arange(label.shape[0])
        class_list = label.squeeze().unique()
        for i in range(class_list.shape[0]):
            c_i = class_list[i]
            idx_i = idx[label.squeeze() == c_i]
            n_i = idx_i.shape[0]
            rand_idx = idx_i[torch.randperm(n_i)]
            train_idx += rand_idx[:label_num_per_class].tolist()
            non_train_idx += rand_idx[label_num_per_class:].tolist()
        train_idx = torch.as_tensor(train_idx)
        non_train_idx = torch.as_tensor(non_train_idx)
        non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
        valid_idx, test_idx = (
            non_train_idx[:valid_num],
            non_train_idx[valid_num : valid_num + test_num],
        )
        print(f"train:{train_idx.shape}, valid:{valid_idx.shape}, test:{test_idx.shape}")
        split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
        return split_idx

    def split(self, graph, idx=None):
        labels = graph.ndata['label']
        if idx is None:
            index = list(range(len(labels)))
            idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                                train_size=self.train_ratio, shuffle=True)
            idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=self.val_ratio/(1-self.train_ratio), shuffle=True)
        else:
            idx_train, idx_valid, idx_test = idx['train'], idx['valid'], idx['test']
        train_mask = torch.zeros([len(labels)]).bool()
        val_mask = torch.zeros([len(labels)]).bool()
        test_mask = torch.zeros([len(labels)]).bool()

        train_mask[idx_train] = 1
        val_mask[idx_valid] = 1
        test_mask[idx_test] = 1

        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask

        return graph

    def load_data(self, args, i):
        path = args.file_paths
        if args.dataset in ('cora', 'citeseer', 'pubmed'):
            if args.dataset.startswith('cora'):
                # data = dgl.data.CoraGraphDataset(raw_dir=path)
                data = Planetoid(root=path, name="cora")[0]
            elif args.dataset.startswith('citeseer'):
                # data = dgl.data.CiteseerGraphDataset(raw_dir=path)
                data = Planetoid(root=path, name="citeseer")[0]
            elif args.dataset.startswith('pubmed'):
                # data = dgl.data.PubmedGraphDataset(raw_dir=path)
                data = Planetoid(root=path, name="pubmed")[0]
            # graph = data[0]
            # args.num_o = data.num_classes
            graph = dgl.graph((data.edge_index[0, :], data.edge_index[1, :]), num_nodes=data.num_nodes)
            graph = dgl.to_bidirected(graph)
            graph.ndata['feat'] = data.x
            graph.ndata['label'] = data.y
            # graph.ndata['train_mask'] = data.train_mask
            # graph.ndata['val_mask'] = data.val_mask
            # graph.ndata['test_mask'] = data.test_mask
            self.indices = self.class_rand_splits(graph.ndata['label'], 20)
            args.num_o = data.y.unique().shape[0]

        elif args.dataset in ('computer', 'photo', 'cs', 'physics'):
            transform = T.NormalizeFeatures()
            if args.dataset.startswith('computer'):
                # data = dgl.data.AmazonCoBuyComputerDataset(raw_dir=path)
                data = Amazon(root=path,
                                 name='Computers', transform=transform)[0]
                self.indices = np.load('./data/amazon-computer_split.npz')
            elif args.dataset.startswith('photo'):
                # data = dgl.data.AmazonCoBuyPhotoDataset(raw_dir=path)
                data = Amazon(root=path,
                                 name='Photo', transform=transform)[0]
                self.indices = np.load('./data/amazon-photo_split.npz')
            elif args.dataset.startswith('cs'):
                # data = dgl.data.CoauthorCSDataset(raw_dir=path)
                data = Coauthor(root=path,
                                 name='CS', transform=transform)[0]
                self.indices = np.load('./data/coauthor-cs_split.npz')
            elif args.dataset.startswith('physics'):
                # data = dgl.data.CoauthorPhysicsDataset(raw_dir=path)
                data = Coauthor(root=path,
                                 name='Physics', transform=transform)[0]
                self.indices = np.load('./data/coauthor-physics_split.npz') 
            # graph = data[0]
            # transform = RowFeatNormalizer(subtract_min=True, node_feat_names=['feat'])
            # graph = transform(graph)
            graph = dgl.graph((data.edge_index[0, :], data.edge_index[1, :]), num_nodes=data.num_nodes)
            graph = dgl.to_bidirected(graph)
            graph.ndata['feat'] = data.x
            graph.ndata['label'] = data.y
            args.num_o = data.y.unique().shape[0]
        
        elif args.dataset.startswith('wikics'):
            # data = dgl.data.WikiCSDataset(raw_dir=path)
            data = WikiCS(root=f'{path}/wikics/')[0]
            graph = dgl.graph((data.edge_index[0, :], data.edge_index[1, :]), num_nodes=data.num_nodes)
            graph = dgl.to_bidirected(graph)
            graph.ndata['feat'] = data.x
            graph.ndata['label'] = data.y
            args.num_o = data.y.unique().shape[0]
            # graph = data[0]
            graph.ndata['train_mask'] = data.train_mask[:, i].bool().contiguous()
            graph.ndata['val_mask'] = torch.logical_or(data.val_mask[:, i].bool().contiguous(), data.stopping_mask[:, i].bool().contiguous())
            graph.ndata['test_mask'] = data.test_mask.bool()
            # args.num_o = data.num_classes

        elif args.dataset in ('squirrel', 'chameleon'):
            if args.dataset.startswith('squirrel'):
                data = np.load('./data/squirrel_filtered.npz')
            elif args.dataset.startswith('chameleon'):
                data = np.load('./data/chameleon_filtered.npz')
            graph = dgl.graph((data['edges'][:, 0],data['edges'][:, 1] ), num_nodes=data['node_features'].shape[0])
            graph = dgl.to_bidirected(graph)
            graph.ndata['feat'] = torch.from_numpy(data['node_features'])
            graph.ndata['label'] = torch.from_numpy(data['node_labels'])
            graph.ndata['train_mask'] = torch.from_numpy(data['train_masks'][i, :]).contiguous()
            graph.ndata['val_mask'] = torch.from_numpy(data['val_masks'][i, :]).contiguous()
            graph.ndata['test_mask'] = torch.from_numpy(data['test_masks'][i, :]).contiguous()
            args.num_o = 5

        elif args.dataset in ('amazon_ratings', 'minesweeper', 'questions', 'roman_empire'):
            # transform = T.NormalizeFeatures()
            data = HeterophilousGraphDataset(name=args.dataset.capitalize(), root=path)[0]
            if args.dataset in ('minesweeper', 'questions'):
                args.metric = 'AUC'
            graph = dgl.graph((data.edge_index[0, :], data.edge_index[1, :]), num_nodes=data.num_nodes)
            graph = dgl.to_bidirected(graph)
            graph.ndata['feat'] = data.x
            graph.ndata['label'] = data.y
            graph.ndata['train_mask'] = data.train_mask[:, i].bool().contiguous()
            graph.ndata['val_mask'] = data.val_mask[:, i].bool().contiguous()
            graph.ndata['test_mask'] = data.test_mask[:, i].bool().contiguous()
            args.num_o = data.y.unique().shape[0]
        
        elif args.dataset in ('ogbn-arxiv', 'ogbn-products'):
            data = DglNodePropPredDataset(name=args.dataset, root=path)
            graph, label = data[0]
            self.indices = data.get_idx_split()
            args.num_o = data.num_classes
            graph.ndata['label'] = label.squeeze()
            if args.dataset == 'ogbn-arxiv':
                graph = dgl.to_bidirected(graph, copy_ndata=True)
            

        elif args.dataset.startswith('pokec'):
            data = scio.loadmat('./data/pokec.mat')
            graph = dgl.graph((data['edge_index'][0, :],data['edge_index'][1, :] ), num_nodes=data['num_nodes'])
            graph = dgl.to_bidirected(graph)
            graph.ndata['feat'] = torch.from_numpy(data['node_feat']).float()
            graph.ndata['label'] = torch.from_numpy(data['label'][0])
            self.indices = np.load('./data/pokec-splits.npy', allow_pickle=True)[i]
            args.num_o = 2

        graph = dgl.add_self_loop(graph)
        graph.ndata['deg'] = graph.out_degrees().float()

        return graph

