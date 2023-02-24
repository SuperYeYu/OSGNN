import torch
import torch.nn as nn
import torch.nn.functional as F
from model import OSGNN
import numpy as np

class OSGNN_GL(nn.Module):
    def __init__(self, input_dim, feat_hid_dim, metapath, dropout,outsize):
        super(OSGNN_GL,self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.metapath = metapath
        self.feat_hid_dim = feat_hid_dim
        self.non_linear = nn.ReLU()
        self.feat_mapping = nn.ModuleList([nn.Linear(m, feat_hid_dim, bias=True) for m in input_dim])
        for fc in self.feat_mapping:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.overall_graph_gen = GraphChannelAttLayer(metapath)

        self.het_graph_encoder_anchor = OSGNN(num_graph=2, hidden_size=feat_hid_dim, out_size=outsize, num_layer=1, dropout=dropout)


    def forward(self, features, G, ADJ, type_mask, norm, agg):

        transformed_features = torch.zeros(type_mask.shape[0], self.feat_hid_dim).to(features[0].device)
        for i, fc in enumerate(self.feat_mapping):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features[i])
        h = transformed_features
        feat_map = self.dropout(h)

        # Subgraph generation of target node
        new_G = self.overall_graph_gen(G, norm[1])
        new_G = new_G.t() + new_G
        new_G = F.normalize(new_G, dim=1, p= norm[0])

        G = [ADJ, new_G]

        logits, h, loss = self.het_graph_encoder_anchor(G, feat_map, agg)
        return logits, h, loss


class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)
    def forward(self, adj_list, norm_2):
        adj_list = torch.stack(adj_list)
        if norm_2 != 0:
            adj_list = F.normalize(adj_list, dim=1, p=norm_2)
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)


