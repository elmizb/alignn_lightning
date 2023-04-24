from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import torch_geometric
import pytorch_lightning as pl
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch_geometric.typing import Adj, OptTensor, PairTensor
from pydantic.typing import Literal
from alignn_pl.config import ALIGNNConfig

class EdgeGatedConv(MessagePassing):
    def __init__(self,
                 input_features: int,
                 output_features: int,
                 aggr: str = 'add',
                 dim = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_features = input_features
        self.output_features = output_features
        # self.gate = nn.Linear(3*output_features, output_features)
        self.src_gate = nn.Linear(output_features, output_features)
        self.dst_gate = nn.Linear(output_features, output_features)
        self.edge_gate = nn.Linear(output_features, output_features)
        self.src_update = nn.Linear(output_features, output_features)
        self.dst_update = nn.Linear(output_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)
        self.reset_parameters()

    def reset_parameters(self):
        # self.gate.reset_parameters()
        self.src_gate.reset_parameters()
        self.dst_gate.reset_parameters()
        self.edge_gate.reset_parameters()
        self.src_update.reset_parameters()
        self.dst_update.reset_parameters()
        self.bn_nodes.reset_parameters()
        self.bn_edges.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
            node_src = node_dst = self.src_gate(x[0])
        else:
            node_src = self.src_gate(x[0])
            node_dst = self.dst_gate(x[1])
        nodes = (node_src, node_dst)
        m = self.edge_updater(edge_index, x=nodes, edge_attr=edge_attr)
        # m = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        sum_sigma = self.propagate(edge_index, x=x, edge_attr=m, for_normalizing=True)
        sum_sigma_h = self.propagate(edge_index, x=x, edge_attr=m)
        out = sum_sigma_h / (sum_sigma + 1e-6)
        out = out + self.src_update(x[0])
        out = F.silu(self.bn_nodes(out))
        out = out + x[1]
        m = F.silu(self.bn_edges(m))
        m = m + edge_attr
        return out, m
    
    def edge_update(self, x_i, x_j, edge_attr) -> Tensor:
        m = x_i + x_j + self.edge_gate(edge_attr)
        return m

    # def edge_update(self, x_i, x_j, edge_attr,) -> Tensor:
    #     print(x_i)
    #     print(x_j)
    #     print(edge_attr)
    #     m = torch.cat([x_i, x_j, edge_attr], dim=-1)
    #     # m = self.gate(m)
    #     return m

    def message(self, x_i, x_j, edge_attr, for_normalizing=False):
        sum_sigma = edge_attr.sigmoid()
        if for_normalizing:
            return sum_sigma
        else:
            return sum_sigma * self.dst_update(x_j)
    
    # def message(self, x_i, x_j, edge_attr, for_normalizing=False,):
    #     sum_sigma = edge_attr.sigmoid()
    #     if for_normalizing:
    #         return sum_sigma
    #     else:
    #         return sum_sigma * self.dst_update(x_j)
    
    def __repr__(self) -> str:
            return f'{self.__class__.__name__}({self.input_features}, {self.output_features})'
    
class ALIGNNConv(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.node_update = EdgeGatedConv(input_features, output_features)
        self.edge_update = EdgeGatedConv(output_features, output_features)

    def forward(
        self,
        g: torch_geometric.data.Data,
        lg: torch_geometric.data.Data,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        ):
        """Node and Edge updates for ALIGNN layer.
        x: node input features ; atomic features
        y: edge input features ; bond length
        z: edge pair input features ; angles
        """
        x, m = self.node_update(x, g.edge_index, y)
        y, z = self.edge_update(m, lg.edge_index, z)

        return x, y, z
    
class MLPLayer(torch.nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, input_features: int, output_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_features, output_features),
            nn.BatchNorm1d(output_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)
    
class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )

class ALIGNN(pl.LightningModule):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.classification = config.classification

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features,),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1, vmax=1.0, bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(config.hidden_features, config.hidden_features,)
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedConv(
                    config.hidden_features, config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = global_mean_pool
        self.fc = nn.Linear(config.hidden_features, config.output_features)
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

        self.metrics = config.metrics
        self.optimizer = config.optimizer
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.scheduler = config.scheduler
        self.epochs = config.epochs

    def forward(
        self, 
        g: Union[Tuple[torch_geometric.data.Data, torch_geometric.data.Data], 
                 torch_geometric.data.Data]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            g, lg = g

            # angle features (fixed)
            z = self.angle_embedding(lg.edge_attr.squeeze(1))

        # initial node features: atom feature network...
        x = self.atom_embedding(g.x["atom_features"])

        # initial bond features
        bondlength = torch.norm(g.edge_attr, dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(x, g.edge_index, y)

        # norm-activation-pool-classify
        h = self.readout(x, batch=None)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)

    def training_step(self, batch, batch_idx):
        g, lg, label = batch
        preds = self.forward((g, lg))
        loss = self.metrics(preds, label)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        g, lg, label = batch
        preds = self.forward((g, lg))
        loss = self.metrics(preds, label)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        try:
            g, lg = batch
        except:
            g, lg, label = batch
        preds = self.forward((g, lg))
        loss = self.metrics(preds, label)
        self.log("test_loss", loss)
        return loss, preds
    
    def predict_step(self, batch, batch_idx):
        try:
            g, lg = batch
        except:
            g, lg, label = batch
        preds = self.forward((g, lg))
        return preds
    
    def configure_optimizers(self):
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        
        if self.scheduler == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        elif self.scheduler == "onecycle":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.learning_rate,
                total_steps=self.epochs,
                steps_per_epoch=1,
                pct_start=0.3
            )
        
        return [optimizer], [lr_scheduler]
    
