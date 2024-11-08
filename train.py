import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import torch
from torch.utils.data import DataLoader
from pre_training import first_brunch, second_brunch, EM_training
import random, os
import warnings
warnings.filterwarnings("ignore")
import os



def pretraining(args):
    first_brunch(args)
    second_brunch(args)

    

def get_train_test_sampler(trainset, valid=0.2):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cutoff', type=int, default=3, metavar='N', help='Max number of nodes in paths (path length +1)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--l2', type=float, default=0.00005, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS', help='batch size')
    parser.add_argument('--num_features', type=int, default=1024, help="input size")
    parser.add_argument('--epochs', type=int, default=1, metavar='E', help='number of epochs')

    parser.add_argument('--source_dataset', type=str, default="LUAD", help="BLCA, BRCA, LGG, LUAD, UCEC")
    parser.add_argument('--source_dataset_dir', type=str, default="/data/ypq/LUAD_Features", help="/data/ypq/BLCA_Features, \
                        /data/ypq/BRCA_Features, /data/ypq/GBMLGG_Features, /data/ypq/LUAD_Features, /data/ypq/UCEC_Features")
    
    parser.add_argument('--target_dataset_dir', type=str, default="/data/ypq/UCEC_Features", help="/data/ypq/LUAD_Features, \
                        /data/ypq/BRCA_Features, /data/ypq/GBMLGG_Features, /data/ypq/LGG_Features, /data/ypq/UCEC_Features")
    parser.add_argument('--target_dataset', type=str, default="UCEC", help="BLCA, BRCA, LGG, LUAD, UCEC")

    parser.add_argument('--k_folds', type=str, default="5foldcv", help="Cross-validation")
    parser.add_argument('--pool_type', type=str, choices=['TopK', 'Edge', 'SAG', 'ASA','GMT'], default='TopK')
    parser.add_argument('--second_model', type=str, default='PathNN', help="PathNN or RandomWalkNN or SampleNN or WL")  
    parser.add_argument('--first_conv_type', type=str, choices=['GCN', 'SAGE', 'GAT', "GIN"], default='GIN')
    parser.add_argument('--conv_type', type=str, choices=['GCN', 'SAGE', 'GAT', 'GIN'], default='GCN')
    parser.add_argument('--second_conv_type', type=str, choices=['pathnn'], default='pathnn')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--e_threshold', type=float, default=0.6)
    parser.add_argument('--m_threshold', type=float, default=0.6)   
    parser.add_argument('--EM_epochs', type=int, default=10)   # coulping times
    parser.add_argument('--m', type=int, default=10) 
    ########### for perturb
    parser.add_argument('--pp', type=str, default="X",
                        help='perturb_position (default: X(feature), H(hidden layer))')
    parser.add_argument('--modal', type=str, default="coattn")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--projection_size', type=int, default=256)
    parser.add_argument('--prediction_size', type=int, default=4)
    parser.add_argument('--delta', type=float, default=8e-1)
    parser.add_argument('--hidden_dim', type=int, default=3)
    parser.add_argument('--OOM', type=int, default=4096)
    parser.add_argument('--step_size', type=float, default=8e-3)
    parser.add_argument('--k_neighbor', type=float, default=2048)
    parser.add_argument('--time', type=float, default=250, help="surv time")
    args = parser.parse_args()
    print(args)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    set_seed(args.seed)
    print(f'source {args.source_dataset} -> target {args.target_dataset}')


    pretraining(args)
    EM_training(args)



