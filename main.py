import numpy as np
from typing import List

from DataLoading import DataLoader
from utils import train, train_lg, set_seed
import torch
import argparse

parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--cuda', type=int, default=0,
                    help='Which GPU to run on (-1 for using CPU, 9 for not specifying which GPU to use.)')
parser.add_argument('--file_paths', type=str, default='./data/')
parser.add_argument('--dataset', type=str, default='cs',
                        help="Dataset for this model (squirrel/chameleon/amazon_ratings/roman_empire/minesweeper/questions"
                             "cora/citeseer/pubmed/computer/photo/cs/physics/wikics"
                             "ogbn-proteins/ogbn-arxiv/ogbn-products/pokec)")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--L', type=int, default=2)
parser.add_argument('--Delta', type=int, default=0)
parser.add_argument('--K', type=int, default=3)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--num_o', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--run', type=int, default=5)
parser.add_argument('--patience', type=int, default=1000)
parser.add_argument('--indropout', type=float, default=0.4)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--weight_decay', type=float, default=0.0000)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=550000)#262144
parser.add_argument('--metric', type=str, default='Acc', help='[Acc, AUC]')
parser.add_argument('--GNN', type=str, default='gcn', help='[gcn, sage, gat]')
parser.add_argument('--norm', type=str, default=None, help='[bn, ln]')
parser.add_argument('--res', action='store_true')
parser.add_argument('--mlp_in', action='store_true')
parser.add_argument('--dot', action='store_true')
parser.add_argument('--mode', type=str, default='full', help='[full, low]')
args = parser.parse_args()
# args.argv = sys.argv

if torch.cuda.is_available():
    if args.cuda == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = args.cuda
        print('using device', device_id, torch.cuda.get_device_name(device_id))

args.device = torch.device(f"cuda:{device_id}" if args.cuda >= 0 else "cpu")

set_seed(args.seed)


def main(state=1):

    if state:
        final_metric = []
        if args.dataset in ('cs', 'physics', 'computer', 'photo', 'cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'ogbn-products'):
            ds = DataLoader(args)
        for i in range(args.run):
            if args.dataset not in ('cs', 'physics', 'computer', 'photo', 'cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'ogbn-products'):
                ds = DataLoader(args, i)
            print(args)
            if args.dataset not in ('ogbn-products', 'pokec'):
                model, tmetric = train(ds, args)
            else:
                model, tmetric = train_lg(ds, args)
            final_metric.append(tmetric)
            # torch.save(model.state_dict(), './model/YC_7_l.pt')
        if args.metric == 'Acc':
            print('Acc-mean: {:.2f}, Acc-std: {:.2f}'.format(100 * np.mean(final_metric), 100 * np.std(final_metric)))
        else:
            print('AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_metric), 100 * np.std(final_metric)))
        print(final_metric)
        with open('result.txt', 'a+') as f:
            f.write(f'{args}\n')
            f.write(f'{100 * np.mean(final_metric)}\n')
            f.write(f'{100 * np.std(final_metric)}\n')
            f.write(f'{final_metric}\n')



if __name__ == '__main__':
    main(1)





