from typing import List, Tuple, Sequence, Optional
from collections import defaultdict
import json
import warnings

from pymatgen.core import Structure
import numpy as np
import pandas as pd
import torch
try:
    from torch_geometric.data import Data
except:
    pass
try:
    import dgl
except:
    pass

from alignn_pl.config import ALIGNNConfig

warnings.filterwarnings('ignore')

class StructureDataset(torch.utils.data.Dataset):
    """
    pandas DataFrame, graphs,(+line graphs)
    をまとめてpytorch Datasetに
    """

    def __init__(
        self,
        df: pd.DataFrame,
        graphs,
        target: str,
        is_test: bool = False, 
        transform=None,
        line_graphs=None,
        classification=False,
        id_tag="id",
    ):
        self.df = df
        self.graphs = graphs.reset_index(drop=True)
        self.line_graphs = line_graphs.reset_index(drop=True)
        self.ids = self.df[id_tag]
        self.transform = transform
        self.is_test = is_test

        if not is_test:
            self.target = target
            # self.labels = self.df[target]
            self.labels = torch.tensor(self.df[target].to_numpy()).type(
                torch.get_default_dtype()
            )
        if classification:
            self.labels = self.labels.view(-1).long()
            print("Classification dataset.", self.labels)
        
        self.prepare_batch = prepare_dgl_batch
        if line_graphs.any():
            self.prepare_batch = prepare_line_graph_batch

    def __len__(self):
        """Get length."""
        return self.ids.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]

        if self.transform:
            g = self.transform(g)

        if not self.is_test:
            label = self.labels[idx]
            if self.line_graphs.any():
                return g, self.line_graphs[idx], label
            else:
                return g, label

        else:
            if self.line_graphs.any():
                return g, self.line_graphs[idx]
            else:
                return g
            
    def setup_standardizer(self, ids):
        """Atom-wise feature standardization transform."""
        x = torch.cat(
            [
                g.x["atom_features"]
                for idx, g in enumerate(self.graphs)
                if idx in ids
            ]
        )
        self.atom_feature_mean = x.mean(0)
        self.atom_feature_std = x.std(0)

        self.transform = Standardize(
            self.atom_feature_mean, self.atom_feature_std
        )
    
    @staticmethod
    def collate(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)

    @staticmethod
    def collate_line_graph(
        samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, torch.tensor(labels)

def structure_dgl_multigraph(
    structure: Structure = None,
    cutoff:float = 8.0,
    max_neighbors:int = 12,
    atom_features:str = "cgcnn",
    compute_line_graph: bool = True    
):
    edges = nearest_neighbor_edges(
        structure=structure,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )
    
    u, v, r = build_undirected_edgedata(structure, edges)

    sps_features = []
    for element in structure.species:
        if atom_features == "cgcnn":
            feat = list(cgcnn_node_attribute(element))

        elif atom_features == "basic":
            feat = basic_node_attribute(element)
        
        sps_features.append(feat)
    sps_features = np.array(sps_features)
    node_features = torch.tensor(sps_features).type(
        torch.get_default_dtype()
    )
    g = dgl.graph((u, v))
    g.ndata["atom_features"] = node_features
    g.ndata["lattice_mat"] = torch.tensor(
        [structure.lattice.matrix for i in range(structure.num_sites)]
    )
    g.edata["r"] = r
    g.ndata["V"] = torch.tensor(
        [structure.lattice.volume for i in range(structure.num_sites)]
    )

    if compute_line_graph:
        # construct atomistic line graph
        # (nodes are bonds, edges are bond pairs)
        # and add bond angle cosines as edge features
        lg = g.line_graph(shared=True)
        lg.apply_edges(compute_bond_cosines_dgl)
        return g, lg
    else:
        return g

def structure_pyg_multigraph(
    structure: Structure = None,
    cutoff:float = 8.0,
    max_neighbors:int = 12,
    atom_features:str = "cgcnn",
    compute_line_graph: bool = True
):
    """pymatgen.core.Structure からgraph,(line_graph)を作成する

    Args:
        structure (Structure, optional): 
            pymatgen.core.Structure. Defaults to None.
        cutoff (float, optional):
            cutoff for the distance of neighboring atoms. Defaults to 8.0.
        max_neighbors (int, optional): 
            max number of atoms to include graph. Defaults to 12.
        atom_features (str, optional): 
            the method to put the features on atom node
            "cgcnn": one-hot encorded atom features, it needs atom_init.json
            "basic": float values obtained from parameters in pymatgen.core.Element
            Defaults to "cgcnn".
        compute_line_graph (bool, optional): 
            Defaults to True.

    Returns:
        torch_geometric.data.Data : graph data in torch_geometric style
    """
    edges = nearest_neighbor_edges(
        structure=structure,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )
    
    u, v, r = build_undirected_edgedata(structure, edges)

    sps_features = []
    for element in structure.species:
        if atom_features == "cgcnn":
            feat = list(cgcnn_node_attribute(element))

        elif atom_features == "basic":
            feat = basic_node_attribute(element)
        
        sps_features.append(feat)
    sps_features = np.array(sps_features)
    node_features = torch.tensor(sps_features).type(
        torch.get_default_dtype()
    )

    g = Data(edge_index=torch.vstack((u, v)))
    g.num_nodes = len(node_features)
    g.x = {}
    g.x["atom_features"] = node_features

    # lattice matrix, volume of the input structure. it is not used for alignn.
    g.x["lattice_mat"] = torch.tensor(
        [structure.lattice.matrix for i in range(structure.num_sites)]
    )
    g.x["V"] = torch.tensor(
        [structure.lattice.volume for i in range(structure.num_sites)]
    )
    g.edge_attr = r

    if compute_line_graph:
        # transform = LineGraph()
        # lg = transform(g.clone())
        # lg.edge_attr = compute_bond_cosines(lg)
        lg = gen_line_graph(g)
        return g, lg
    else:
        return g

def nearest_neighbor_edges(
    structure: Structure = None,
    cutoff: float = 8,
    max_neighbors: int = 12,
) -> dict :
    """
    From pymatge.core.Structure return the dict of graph edges.
    The original function has some arguments like use_canonize.
    This simplified code is written for the case use_canonize=True.

    Args:
        structure (Structure, optional): Defaults to None.
        cutoff (float, optional): Defaults to 8.
        max_neighbors (int, optional): Defaults to 12.

    Returns:
        dict: 
            key : (source site id, distination site id)
            value : distination image from (0, 0, 0)
    """
    
    all_neighbors = structure.get_all_neighbors(r=cutoff)

    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)
    # attempt = 0
    # if sufficient atoms are not in all_neighbors, extend r_cutoff and get neighbors again.
    while min_nbrs < max_neighbors:
        # print("extending cutoff radius!", attempt, cutoff)
        lat = structure.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            cutoff = max(lat.a, lat.b, lat.c)
        else:
            cutoff = 2 * cutoff
        all_neighbors = structure.get_all_neighbors(r=cutoff)
        min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)
        # attempt += 1

    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):
        
        # sort by the distance from origin site : neighborlist[2]
        neighborlist = sorted(neighborlist, key=lambda x: x[1]) 
        # store the value of each site
        # the order of values in neighbor is different between jarvis and pymatgen
        distances = np.array([nbr[1] for nbr in neighborlist])
        ids = np.array([nbr[2] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # keep all edges out to the neighbor shell
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]

        # with using canonize_edge function, 
        # the distination images are reduced to the shape measured from (0, 0, 0).
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            edges[(src_id, dst_id)].add(dst_image)
    return edges

def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
) -> Tuple:

    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image
    
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image

def build_undirected_edgedata(
    structure: Structure = None,
    edges: dict = {}
) -> Tuple:
    """
    it does
        compute the distance between src and dst sites
            by fractional position in reduced cell + image(it means cell position in supercell)
        transform one-way input edges to round trip edges(undirected style).

    Args:
        structure (Structure, optional): Defaults to None.
        edges (dict, optional): Defaults to {}.

    Returns:
        Tuple: 
            u : source site ids in edge list
            v : distination site ids in edge list
            r : distances between src and dst sites in cartesian style 
    """

    u, v, r = [], [], []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = structure.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = structure.lattice.get_cartesian_coords(
                dst_coord - structure.frac_coords[src_id]
            )
            # add edges for both directions
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
    u, v, r = (np.array(x) for x in (u, v, r))
    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r).type(torch.get_default_dtype())

    return u, v, r           

def cgcnn_node_attribute(element):
    Z = str(element.Z)
    with open(ALIGNNConfig.path_cgcnn_atominit, "r") as f:
        # For alternative features use
        # get_digitized_feats_hot_encoded()
        i = json.load(f)
        try:
            return i[Z]
        except KeyError:
            print(f"warning: could not load CGCNN features for {element}")
            print("Setting it to max atomic number available here, 103")
            # TODO Check for the error in oqmd_3d_no_cfid dataset
            # return i['Lr']
            return i["100"]

def basic_node_attribute(element):
    feats = [
        "Z", "X", 
        "atomic_radius_calculated", 
        "ionization_energy",
        "group", "row",
        "electron_affinity"
        ]
    tmp = []
    for f in feats:
        tmp.append(getattr(element, f))
    return tmp

def compute_bond_cosines(graph: Data) -> torch.tensor:
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = graph.x[graph.edge_index[0]]
    r2 = graph.x[graph.edge_index[1]]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine.unsqueeze(1)

def compute_bond_cosines_dgl(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))
    # print (r1,r1.shape)
    # print (r2,r2.shape)
    # print (bond_cosine,bond_cosine.shape)
    return {"h": bond_cosine}

class Standardize(torch.nn.Module):
    """Standardize atom_features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Register featurewise mean and standard deviation."""
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, data: Data):
        """Apply standardization to atom_features."""
        h = data.x
        data.x = (h - self.mean) / self.std
        return data
    
def gen_line_graph(g:Data)-> Data:
    u, v = g.edge_index
    l_src, l_dst = zip(*[(i, j) 
                      for i, dst in enumerate(v) 
                      for j, src in enumerate(u) 
                      if dst == src and i != j])
    l_src = torch.tensor(l_src)
    l_dst = torch.tensor(l_dst)
    lg = Data(edge_index=torch.vstack((l_src, l_dst)))
    lg.x = g.edge_attr
    lg.edge_attr = compute_bond_cosines(lg)
    return lg

def plot_graph(g:Data)-> None:
    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx 
    G = to_networkx(g)
    pos = nx.random_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color = 'r', node_size = 100, alpha = 1)
    ax = plt.gca()
    for e in G.edges:
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                    ),
                                    ),
                    )
    plt.axis('off')
    plt.show()

def prepare_dgl_batch(
    batch: Tuple[dgl.DGLGraph, torch.Tensor], device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    g, t = batch
    batch = (
        g.to(device, non_blocking=non_blocking),
        t.to(device, non_blocking=non_blocking),
    )

    return batch


def prepare_line_graph_batch(
    batch: Tuple[Tuple[dgl.DGLGraph, dgl.DGLGraph], torch.Tensor],
    device=None,
    non_blocking=False,
):
    """Send line graph batch to device.

    Note: the batch is a nested tuple, with the graph and line graph together
    """
    g, lg, t = batch
    batch = (
        (
            g.to(device, non_blocking=non_blocking),
            lg.to(device, non_blocking=non_blocking),
        ),
        t.to(device, non_blocking=non_blocking),
    )

    return batch