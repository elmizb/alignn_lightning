from typing import Literal
from tqdm import tqdm
import torch
try:
    from torch_geometric.loader import DataLoader as DataLoader_pyg
except:
    from torch.utils.data import DataLoader
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor
import pytorch_lightning as pl

from alignn_pl.graph_vanilla import StructureDataset, structure_pyg_multigraph, structure_dgl_multigraph
from alignn_pl.config import ALIGNNConfig as config

tqdm.pandas()

class ALIGNNDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_dir: str = "path/to/dir",
            graph_fmt:Literal["dgl", "pyg"] = "dgl", 
            batch_size: int=config.batch_size,
            num_workers: int=config.num_workers,
            pin_memory: bool=config.pin_memory,
            shuffle: bool=config.shuffle
            ):
        super().__init__()
        self.data_dir = data_dir
        self.graph_fmt = graph_fmt
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
    
    def prepare_data(
            self, 
            df_train: pd.DataFrame,
            df_valid: pd.DataFrame,
            df_test: pd.DataFrame,
            use_pandarallel: bool=False,
            ):
        if use_pandarallel:
            from pandarallel import pandarallel
            pandarallel.initialize(progress_bar=True)
        get_structure_dataset(
            df_train, 
            save=True, 
            format=self.graph_fmt, 
            filename=self.data_dir+"train.data", 
            use_pandarallel=use_pandarallel
            )
        get_structure_dataset(
            df_valid, 
            save=True, 
            format=self.graph_fmt, 
            filename=self.data_dir+"valid.data", 
            use_pandarallel=use_pandarallel
            )
        get_structure_dataset(
            df_test, 
            is_test=False, 
            save=True, 
            format=self.graph_fmt,
            filename=self.data_dir+"test.data", 
            use_pandarallel=use_pandarallel
            )
        
    def setup(self, stage: str, minmax_scale: bool=False, standard_scale: bool=False):
        if stage == "fit":
            self.dataset_train = torch.load(self.data_dir+"train.data")
            self.dataset_valid = torch.load(self.data_dir+"valid.data")
            try:
                if self.dataset_train.line_graphs.any():
                    self.collate_fn = self.dataset_train.collate_line_graph
                else:
                    self.collate_fn = self.dataset_train.collate
            except:
                pass

            if minmax_scale:
                labels_train = self.dataset_train.labels
                label_min = labels_train.min()
                label_max = labels_train.max()
                labels_train = (labels_train - label_min) / (label_max - label_min)
                labels_valid = (self.dataset_valid.labels - label_min) / (label_max - label_min)
                self.dataset_train.labels = labels_train
                self.dataset_valid.labels = labels_valid
                torch.save(label_max, self.data_dir+"label_max.data")
                torch.save(label_min, self.data_dir+"label_min.data")

            if standard_scale:
                labels_train = self.dataset_train.labels
                label_mean = labels_train.mean()
                label_std = labels_train.std()
                labels_train = (labels_train - label_mean) / label_std
                labels_valid = (self.dataset_valid.labels - label_mean) / label_std
                self.dataset_train.labels = labels_train
                self.dataset_valid.labels = labels_valid
                torch.save(label_mean, self.data_dir+"label_mean.data")
                torch.save(label_std, self.data_dir+"label_std.data")

        if stage == "test":
            self.dataset_test = torch.load(self.data_dir+"test.data")
            try:
                if self.dataset_test.line_graphs.any():
                    self.collate_fn = self.dataset_test.collate_line_graph
                else:
                    self.collate_fn = self.dataset_test.collate
            except:
                pass

    def train_dataloader(self):

        if self.graph_fmt == "dgl":
            return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        else:
            return DataLoader_pyg(
                self.dataset_train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
    
    def val_dataloader(self):
        
        if self.graph_fmt == "dgl":
            return DataLoader(
                self.dataset_valid,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        else:
            return DataLoader_pyg(
                self.dataset_valid,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
    
    def test_dataloader(self, batch_size=1):
        
        if self.graph_fmt == "dgl":
            return DataLoader(
                self.dataset_test,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        else:
            return DataLoader_pyg(
                self.dataset_test,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        

def get_structure_dataset(
    df: pd.DataFrame,
    format: Literal["pyg", "dgl"]="pyg",
    is_test: bool=False,
    id_tag: str=config.id_tag,
    target: str=config.target,
    structure_tag: str=config.structure_format,
    atom_features: str=config.atom_features,
    compute_line_graph: bool=config.compute_line_graph,
    cutoff: float=config.cutoff, 
    max_neighbors: int=config.max_neighbors,
    classification: bool=False,
    save: bool=config.save_dataset,
    filename: str=config.filename,
    use_pandarallel: bool=False
):
    """
    df[structure_tag] -> torch.utils.data.Dataset
        ; cif data in dataframe -> dataset of graphs with atoms and bonds 
    
    Args:
        df (pd.DataFrame): _description_
        is_test (bool, optional): it defines use label data or not. Defaults to False.
        batch_size (int, optional): Defaults to 64.
        num_workers (int, optional): Defaults to 4.
        id_tag (str, optional): name of index column in df. Defaults to "id".
        target (str, optional): name of label column in df. Defaults to "".
        structure_tag (str, optional): Defaults to "cif". cif is only supported.
        atom_features (str, optional): Defaults to "cgcnn". cgcnn and basic are supported.
        compute_line_graph (bool, optional): Defaults to True.
        cutoff (float, optional): Defaults to 8.0.
        max_neighbors (int, optional): Defaults to 12.
        classification (bool, optional): Defaults to False.
        save_dataloader (bool, optional): Defaults to False.
        filename (str, optional): Defaults to "".
    """
    
    def str_to_graph(cif_string):
        if structure_tag == "cif":
            crystal = Structure.from_str(cif_string, fmt="cif")
        elif structure_tag == "atoms":
            from jarvis.core.atoms import Atoms
            atoms = Atoms.from_dict(cif_string)
            crystal = JarvisAtomsAdaptor.get_structure(atoms)
        if format == "pyg":
            graph = structure_pyg_multigraph(
                crystal,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                atom_features=atom_features,
                compute_line_graph=compute_line_graph
                )
        elif format == "dgl":
            graph = structure_dgl_multigraph(
                crystal,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                atom_features=atom_features,
                compute_line_graph=compute_line_graph
                )
        return graph
    
    if use_pandarallel:
        graphs = df[structure_tag].parallel_apply(lambda x: pd.Series(str_to_graph(x)).reset_index(drop=True))
    else:
        graphs = df[structure_tag].progress_apply(lambda x: pd.Series(str_to_graph(x)).reset_index(drop=True))

    if compute_line_graph:
        line_graph = graphs[1]
        graphs = graphs[0]
    else:
        line_graph = None

    dataset = StructureDataset(
        df,
        graphs=graphs,
        target=target,
        is_test=is_test,
        line_graphs=line_graph,
        classification=classification,
        id_tag=id_tag
    )
    
    if save:
        torch.save(dataset, filename)
    return dataset


