python main.py --dataset computer --L 2 --K 5 --hidden_size 512 --lr 0.001 --indropout 0.5 --dropout 0.5 --weight_decay 0.00005 --norm ln --run 3
python main.py --dataset photo --L 5 --K 2 --hidden_size 128 --lr 0.001 --indropout 0.5 --dropout 0.5 --weight_decay 0.00005 --norm ln --res --mlp_in --GNN gat --run 3
python main.py --dataset cs --L 2 --K 3 --hidden_size 256 --lr 0.001 --indropout 0.4 --dropout 0.4 --weight_decay 0.000 --norm ln --res --GNN gat --run 3
python main.py --dataset physics --L 2 --K 3 --hidden_size 256 --lr 0.001 --indropout 0.5 --dropout 0.6 --weight_decay 0.0005 --norm ln --res --run 3
python main.py --dataset wikics --L 2 --K 7 --hidden_size 128 --lr 0.001 --indropout 0.6 --dropout 0.7 --weight_decay 0.000 --norm ln --GNN gat --run 3

python main.py --dataset amazon_ratings --L 3 --K 1 --hidden_size 512 --lr 0.001 --indropout 0.1 --dropout 0.5 --weight_decay 0.000 --norm bn --res --GNN gat --dot --run 3
python main.py --dataset roman_empire --L 9 --K 3 --hidden_size 512 --lr 0.001 --indropout 0.3 --dropout 0.3 --weight_decay 0.000 --norm bn --res --mlp_in --run 3
python main.py --dataset minesweeper --L 17 --K 4 --hidden_size 64 --lr 0.01 --indropout 0.1 --dropout 0.1 --weight_decay 0.000 --norm bn --res --GNN gat --run 3
python main.py --dataset questions --L 13 --K 5 --hidden_size 512 --lr 0.00003 --indropout 0.3 --dropout 0.3 --weight_decay 0.000 --mlp_in --res --dot --run 3

python main.py --dataset ogbn-arxiv --L 6 --K 3 --hidden_size 512 --lr 0.0005 --indropout 0.0 --dropout 0.5 --weight_decay 0.0005 --norm bn --res --run 3
python main.py --dataset ogbn-products --L 5 --K 3 --hidden_size 256 --batch_size 262144 --lr 0.003 --indropout 0.0 --dropout 0.5 --weight_decay 0.000 --norm ln --GNN sage --mlp_in --dot --patience 500 --run 3
python main.py --dataset pokec --L 9 --K 3 --hidden_size 256 --batch_size 550000 --lr 0.0005 --indropout 0.0 --dropout 0.2 --weight_decay 0.000 --norm bn --res --run 5