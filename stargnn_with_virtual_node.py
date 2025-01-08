import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class StarGNNWithVirtualNode(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_virtual_nodes=1):
        super(StarGNNWithVirtualNode, self).__init__()

        # 图注意力层
        self.gat_layer1 = GATConv(in_features, hidden_features, heads=2, concat=True)
        self.gat_layer2 = GATConv(hidden_features * 2, out_features, heads=1, concat=False)

        # 虚拟节点特征初始化
        self.num_virtual_nodes = num_virtual_nodes
        self.virtual_node_emb = nn.Parameter(torch.randn(num_virtual_nodes, in_features))

        # 跳层聚合的线性层
        self.jump_layer = nn.Linear(in_features + hidden_features * 2 + out_features, out_features)

    def forward(self, graph, node_features, edge_index):
        """
        Parameters:
        - graph: 图数据（用于存储节点和边信息）。
        - node_features: 节点特征矩阵 (num_nodes, in_features)。
        - edge_index: 边索引矩阵 (2, num_edges)，每列表示一个边（起点，终点）。
        """
        # Step 1: 动态添加虚拟节点
        virtual_nodes = self.virtual_node_emb.expand(graph.num_nodes, -1, -1)  # 动态扩展
        augmented_features = torch.cat([node_features, virtual_nodes], dim=0)

        # Augment graph edges to connect virtual nodes
        num_nodes = graph.num_nodes
        virtual_edges = self._add_virtual_edges(num_nodes, self.num_virtual_nodes)
        augmented_edge_index = torch.cat([edge_index, virtual_edges], dim=1)

        # Step 2: 图注意力层计算
        hidden_features = F.elu(self.gat_layer1(augmented_features, augmented_edge_index))
        output_features = self.gat_layer2(hidden_features, augmented_edge_index)

        # Step 3: 跳层聚合 (combine original, hidden, and output)
        combined_features = torch.cat([node_features, hidden_features[:num_nodes], output_features[:num_nodes]], dim=1)
        final_output = self.jump_layer(combined_features)

        return final_output

    def _add_virtual_edges(self, num_nodes, num_virtual_nodes):
        """
        Adds edges between real nodes and virtual nodes.
        """
        real_node_indices = torch.arange(0, num_nodes, dtype=torch.long)
        virtual_node_indices = torch.arange(num_nodes, num_nodes + num_virtual_nodes, dtype=torch.long)

        # Fully connect real nodes to virtual nodes
        virtual_edges = torch.stack([real_node_indices.repeat(num_virtual_nodes),
                                     virtual_node_indices.repeat_interleave(num_nodes)], dim=0)
        return virtual_edges