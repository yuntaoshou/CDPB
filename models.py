import torch.nn as nn
from torch.nn import Sequential as Seq
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch_geometric.nn as gnn
from torch_scatter import scatter_add
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch_geometric.nn import global_add_pool
from typing import Callable, Union
from torch import Tensor
from torch_geometric.typing import OptPairTensor
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def create_knn_edges(features, k_features, k=50, num_dim=1024):
    features = features[:k_features, :]
    features_reshaped = features.reshape(-1, num_dim).detach().cpu()
    knn = NearestNeighbors(n_neighbors=k+1)  # k+1因为包括自身
    knn.fit(features_reshaped)

    # 获取K近邻索引
    distances, indices = knn.kneighbors(features_reshaped)

    # 构建边索引
    edges = []
    for i in range(indices.shape[0]):
        for j in range(1, k+1):  # 跳过自身
            edges.append((i, indices[i, j]))

    edges = np.array(edges).T  # 转置为边索引格式
    edge_index = torch.tensor(edges, dtype=torch.long)
    return features_reshaped, edge_index


class GNN(nn.Module):
    def __init__(self, args, num_features, num_classes, conv_type='GIN', pool_type='TopK', emb=True, perturb_position='H') -> None:
        super().__init__()
        self.args = args
        self.device = args.device
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = args.projection_size
        self.pooling_ratio = 0.5
        self.conv_type = conv_type
        self.pool_type = pool_type
        self.perturb_position = perturb_position

        self.embedding = nn.Linear(self.num_features, self.hidden_dim) if emb else None


        # Define convolutional layers
        if self.conv_type == 'GCN':
            self.conv1 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
        elif conv_type == 'SAGE':
            self.conv1 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
        elif conv_type == 'GAT':
            self.conv1 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
            self.conv2 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
            self.conv3 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
        elif conv_type == 'GIN':
            self.conv1 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
            self.conv2 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
            self.conv3 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
        else:
            raise ValueError("Invalid conv_type: %s" % conv_type)

        # Define Pooling layers
        if pool_type == 'TopK':
            self.pool1 = gnn.TopKPooling(self.hidden_dim, self.pooling_ratio)
            self.pool2 = gnn.TopKPooling(self.hidden_dim, self.pooling_ratio)
            self.pool3 = gnn.TopKPooling(self.hidden_dim, self.pooling_ratio)
        elif pool_type == 'SAG':
            self.pool1 = gnn.SAGPooling(self.hidden_dim, self.pooling_ratio)
            self.pool2 = gnn.SAGPooling(self.hidden_dim, self.pooling_ratio)
            self.pool3 = gnn.SAGPooling(self.hidden_dim, self.pooling_ratio)
        elif pool_type == 'Edge':
            self.pool1 = gnn.EdgePooling(self.hidden_dim)
            self.pool2 = gnn.EdgePooling(self.hidden_dim)
            self.pool3 = gnn.EdgePooling(self.hidden_dim)
        elif pool_type == 'ASA':
            self.pool1 = gnn.ASAPooling(self.hidden_dim)
            self.pool2 = gnn.ASAPooling(self.hidden_dim)
            self.pool3 = gnn.ASAPooling(self.hidden_dim)
        else:
            raise ValueError("Invalid pool_type %s" % pool_type)

        self.bn1 = nn.LayerNorm(self.hidden_dim)
        self.bn2 = nn.LayerNorm(self.hidden_dim)
        self.bn3 = nn.LayerNorm(self.hidden_dim)
        self.bn4 = nn.LayerNorm(self.hidden_dim * 2)

        # Define Linear Layers
        # self.linear1 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.linear1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)      # 改了
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.linear3 = nn.Linear(self.hidden_dim, self.num_classes)

        # Define activation function
        self.relu = F.leaky_relu
        self.dropout = nn.Dropout(args.dropout)


    def forward(self, data, perturb=None):
        x, edge_index = create_knn_edges(data, self.args.k_neighbor)
        if self.args.cuda:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(self.device)
            x, edge_index = x.to(self.device), edge_index.to(self.device)
        else:
            batch = torch.zeros(x.shape[0], dtype=torch.long)
        if x.shape[-1] < self.num_features:
            x = torch.cat((x, torch.zeros(x.shape[0], self.num_features-x.shape[-1]).to(self.args.device)),dim=-1).to(self.args.device)
        edge_attr = None
        if perturb is None:
            perturb_shape = (x.shape[0], self.hidden_dim)
            perturb = torch.FloatTensor(*perturb_shape).uniform_(-self.args.delta, self.args.delta).to(self.args.device)
            perturb.requires_grad_()

        if self.perturb_position == 'X' and perturb is not None:
            x = self.embedding(x) + perturb
        else:
            x = self.embedding(x)

        ############################### 如果是在深层加扰动，加在第一层上面
        if self.perturb_position == 'H' and perturb is not None:
            x = self.bn1(self.conv1(x, edge_index, edge_attr))
            x += perturb
            x = self.dropout(self.relu(x, negative_slope=0.1))
        else:
            x = self.dropout(self.relu(self.bn1(self.conv1(x, edge_index, edge_attr)), negative_slope=0.1))


        if self.pool_type == 'Edge':
            x, edge_index, batch, _ = self.pool1(x, edge_index, batch=batch)
        elif self.pool_type == 'ASA':
            x, edge_index, _, batch, _ = self.pool1(x, edge_index, batch=batch)
        elif self.pool_type == 'GMT':
            pass
        elif self.pool_type == 'TopK':
            x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, batch=batch)



        x1 = torch.cat([gnn.global_max_pool(x, batch), gnn.global_mean_pool(x, batch)], dim=1)

        x = self.dropout(self.relu(self.bn2(self.conv2(x, edge_index, edge_attr)), negative_slope=0.1))
        if self.pool_type == 'Edge':
            x, edge_index, batch, _ = self.pool1(x, edge_index, batch=batch)
        elif self.pool_type == 'ASA':
            x, edge_index, _, batch, _ = self.pool1(x, edge_index, batch=batch)
        elif self.pool_type == 'GMT':
            pass
        elif self.pool_type == 'TopK':
            x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, batch=batch)


        x2 = torch.cat([gnn.global_max_pool(x, batch), gnn.global_mean_pool(x, batch)], dim=1)

        x = self.dropout(self.relu(self.bn3(self.conv3(x, edge_index, edge_attr)), negative_slope=0.1))
        if self.pool_type == 'Edge':
            x, edge_index, batch, _ = self.pool1(x, edge_index, batch=batch)
        elif self.pool_type == 'ASA':
            x, edge_index, _, batch, _ = self.pool1(x, edge_index, batch=batch)
        elif self.pool_type == 'GMT':
            pass
        elif self.pool_type == 'TopK':
            x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, batch=batch)

        x3 = self.bn4(torch.cat([gnn.global_max_pool(x, batch), gnn.global_mean_pool(x, batch)], dim=1))
        x_g = self.relu(x1, negative_slope=0.1) + \
            self.relu(x2, negative_slope=0.1) + \
            self.relu(x3, negative_slope=0.1)
        feature = self.linear1(x_g)
        x = self.linear3(feature)
        predict = torch.softmax(x, dim=1)
        S = torch.cumprod(1 - predict, dim=1)
        return feature, predict, S


class PathNN(nn.Module):
    """
    Path Neural Networks that operate on collections of paths. Uses 1 LSTM shared across convolutional layers. 
    """
    def __init__(self, args, input_dim, hidden_dim, cutoff, n_classes, device, dropout, residuals = True, encode_distances=False, perturb_position='X'):
        super(PathNN, self).__init__()
        self.input_dim = input_dim
        self.cutoff = cutoff
        self.args = args
        self.device = device
        self.residuals = residuals 
        self.dropout = dropout 
        self.encode_distances = encode_distances
        self.perturb_position = perturb_position

        #Feature Encoder that projects initial node representation to d-dim space
        self.feature_encoder = Sequential(Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), ReLU(),
                                          Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), ReLU())
        conv_class = PathConv

        #1 shared LSTM across layers
        if encode_distances : 
            self.distance_encoder = nn.Embedding(cutoff, hidden_dim)
            self.lstm = nn.LSTM(input_size = hidden_dim * 2, hidden_size = hidden_dim , batch_first=True, bidirectional = False, num_layers = 1, bias = True)
        else : 
            self.lstm = nn.LSTM(input_size = hidden_dim , hidden_size = hidden_dim , batch_first=True, bidirectional = False, num_layers = 1, bias = True)
        
        self.convs = nn.ModuleList([])
        for _ in range(self.cutoff - 1) : 
            bn = nn.LayerNorm(hidden_dim)
            self.convs.append(conv_class(hidden_dim, self.lstm, bn, residuals = self.residuals, dropout = self.dropout))
        self.norm = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
        self.linear1 = Linear(hidden_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, n_classes)

        self.reset_parameters()

    def reset_parameters(self) :

        for c in self.feature_encoder.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        self.lstm.reset_parameters()
        for conv in self.convs : 
            conv.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()     
        if hasattr(self, "distance_encoder") : 
            nn.init.xavier_uniform_(self.distance_encoder.weight.data)
            
    def forward(self, data, perturb = None):

        #Projecting init node repr to d-dim space
        # [n_nodes, hidden_size]
        x, edge_index = create_knn_edges(data, self.args.k_neighbor)
        if self.args.cuda:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(self.device)
            x, edge_index = x.to(self.device), edge_index.to(self.device)
        else:
            batch = torch.zeros(x.shape[0], dtype=torch.long)
        path_2 = edge_index.T.flip(1)
        if x.shape[-1] < self.input_dim:
            x = torch.cat((x, torch.zeros(x.shape[0], self.input_dim - x.shape[-1]).to(self.device)), dim=-1).to(self.device)
        
        if self.perturb_position == 'X' and perturb is not None:
            h = self.feature_encoder(x) + perturb
        else:
            h = self.feature_encoder(x)
                    
        #Looping over layers
        for i in range(self.cutoff-1) :
            if self.encode_distances : 
                #distance encoding with shared distance embedding
                # [n_paths, path_length, hidden_size]
                dist_emb = self.distance_encoder(getattr(data, f"sp_dists_{i+2}"))
            else : 
                dist_emb = None
            # [n_nodes, hidden_size]
            h = self.convs[i](h, path_2, dist_emb)
            if i == 0:
                if self.perturb_position == 'H' and perturb is not None:
                    h += perturb

        #Readout sum function
        h = self.norm(global_add_pool(h, batch))
        x = self.linear2(h) 
        predict = torch.softmax(x, dim=1)
        S = torch.cumprod(1 - predict, dim=1)
        return h, predict, S
    

class PathConv(nn.Module):
    r"""
    The Path Aggregator module that computes result of Equation 2. 
    """
    def __init__(self, hidden_dim, rnn: Callable, batch_norm : Callable, residuals = True, dropout = 0):
        super(PathConv, self).__init__()
        self.rnn = rnn
        self.bn = batch_norm
        self.hidden_dim = hidden_dim
        self.residuals = residuals
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self) : 
        if self.bn is not None : 
            self.bn.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], paths, dist_emb = None):
        
        h = x[paths]
        
        #Add distance encoding if needed
        if dist_emb is not None : 
            # [n_paths, path_length, hidden_size * 2]
            h = torch.cat([h, dist_emb], dim = -1)

        #Dropout applied before input to LSTM 
        h = F.dropout(h, training=self.training, p=self.dropout)
        
        # [1, n_paths, hidden_size]
        _, (h,_) = self.rnn(h)

        #Summing paths representations based on starting node 
        # [n_nodes, hidden_size]
        h = scatter_add(h.squeeze(0), paths[:,-1], dim = 0, out = torch.zeros(x.size(0), self.hidden_dim, device = x.device))

        #Residual connection
        if self.residuals : 
            h = self.bn(h + x)
        else:  
            h = self.bn(h)
            
        #ReLU non linearity as the phi function
        h = F.relu(h)

        return h
    
    
class MNN_GNN(nn.Module):
    def __init__(self, args, num_classes, conv_type='GCN'):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.hidden_dim = args.projection_size
        self.pooling_ratio = 0.5
        self.conv_type = conv_type

        if self.conv_type == 'GCN':
            self.conv1 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
        elif conv_type == 'SAGE':
            self.conv1 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
        elif conv_type == 'GAT':
            self.conv1 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
            self.conv2 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
            self.conv3 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
        elif conv_type == 'GIN':
            self.conv1 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
            self.conv2 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
            self.conv3 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
        elif conv_type == 'GMT':
            self.conv1 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
        else:
            raise ValueError("Invalid conv_type: %s" % conv_type)
        
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        # Define Linear Layers
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim//2)
        self.linear3 = nn.Linear(self.hidden_dim//2, self.num_classes)

        # Define activation function
        self.relu = F.leaky_relu
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, edge_index):
        edge_attr = None
        feature = x
        if torch.cuda.is_available():
            feature = feature.to(self.args.device)
            edge_index = edge_index.to(self.args.device)

        x1 = self.dropout(self.relu(self.conv1(x, edge_index, edge_attr), negative_slope=0.1))
        x = self.relu(x1, negative_slope=0.1)
        x = self.dropout(self.relu(self.bn1(x), negative_slope=0.1))
        x = feature + 0.01 * x
        x = self.relu(self.linear1(x), negative_slope=0.1)

        x = self.linear3(x)
        predict = torch.softmax(x, dim=1)
        S = torch.cumprod(1 - predict, dim=1)
        return predict, S