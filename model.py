import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = HeteroConv({
            edge_type: SAGEConv(in_channels, hidden_channels)
            for edge_type in metadata['edge_types']
        }, aggr='sum')  # Aggregates over edge types

        self.conv2 = HeteroConv({
            edge_type: SAGEConv(hidden_channels, out_channels)
            for edge_type in metadata['edge_types']
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        # First layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}

        # Second layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict
