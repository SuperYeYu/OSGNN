import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from torch.nn.parameter import Parameter
import math

device = torch.device("cuda:0")
def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).to(device) - (1/dim) * torch.ones(dim, dim).to(device)#[3025, 3025]
    K1 = torch.mm(emb1, emb1.t())#[3025, 3025]emb1.shape=[3025, 256]
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)#[3025, 3025]
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))#公式14
    return HSIC

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size, 1, bias=False))
    def forward(self, z, agg):
        # 拼接
        if agg == 'contact':
            beta = torch.Tensor([[1], [0]]).cuda()
            beta = beta.expand((z.shape[0],) + beta.shape)
            z1 = (beta * z).sum(1)
            alpha = torch.Tensor([[0], [1]]).cuda()
            alpha = alpha.expand((z.shape[0],) + alpha.shape)
            z2 = (alpha * z).sum(1)
            z = torch.cat((z1, z2), 1)
        # 注意力
        elif agg == 'att':
            w = self.project(z).mean(0)
            beta = torch.softmax(w, dim=0)
            # print(beta)
            # beta=torch.tensor([[0.5],[0.5]])
            beta = beta.expand((z.shape[0],) + beta.shape)
            z = (beta * z).sum(1)
        # 平均
        elif agg == 'mean':
            beta = torch.tensor([[0.5], [0.5]])
            beta = beta.expand((z.shape[0],) + beta.shape)
            z = (beta * z).sum(1)
        return z


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return F.elu(output + self.bias)
        else:
            return F.elu(output)



class OSGNNLayer(nn.Module):

    def __init__(self, num_graph, in_size, out_size, dropout):
        super(OSGNNLayer, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        for i in range(num_graph):
            if i == 0:
                self.gcn_layers.append(GATConv(in_size, out_size, 1, dropout, dropout, activation=F.elu, allow_zero_in_degree=True))
            else:
                self.gcn_layers.append(GraphConvolution(in_size, out_size))
        self.semantic_attention = SemanticAttention(in_size=out_size)
        self.num_graph = num_graph

    def forward(self, gs, h, agg):
        semantic_embeddings = []
        for i, g in enumerate(gs):
            if i == 0:
                semantic_embeddings.append(self.gcn_layers[0](gs[0], h).flatten(1)[:len(gs[1]), :])
            else:
                semantic_embeddings.append(self.gcn_layers[i](h[:len(g), :], g).flatten(1))
        loss = loss_dependence(semantic_embeddings[0], semantic_embeddings[1], dim=semantic_embeddings[0].shape[0])
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings, agg), loss

class OSGNN(nn.Module):
    def __init__(self, num_graph, hidden_size, out_size, num_layer, dropout):
        super(OSGNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(OSGNNLayer(num_graph, hidden_size, 32, dropout))
        for l in range(1, num_layer):
            self.layers.append(OSGNNLayer(num_graph, hidden_size, hidden_size, dropout))
        self.predict = nn.Linear(32*2, out_size)

    def forward(self, g, h, agg):
        a = 0
        h1 = h[:len(g[1]), :]
        loss = 0
        for gnn in self.layers:
            h, l = gnn(g, h, agg)
            loss = loss + l
        # 残差链接
        h = (1 - a) * h + a * h1
        return self.predict(h), h, loss