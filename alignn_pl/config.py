from typing import Optional, Union, Literal
import os
import torch.nn.functional as F

FEATURESET_SIZE = {"basic": 11, "atomic_number": 1, "cfid": 438, "cgcnn": 92}

class ALIGNNConfig:
    # alignn layer configuration
    classification = False
    alignn_layers: int = 2
    gcn_layers: int = 2
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    egc_input_features: int = 768
    hidden_features: int = 256
    output_features: int = 1
    link: Literal["identity", "log", "logit"] = "identity"

    # data configuration
    target: str = "formation_energy_peratom"
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn"] = "cgcnn"
    path_cgcnn_atominit = "./alignn_pyg/atom_init.json"
    neighbor_strategy: Literal["k-nearest", "voronoi"] = "k-nearest"
    id_tag: Literal["jid", "id"] = "jid"
    structure_format: str = "atoms"#"cif"
    compute_line_graph: bool = True 
    batch_size: int = 64
    pin_memory: bool = True
    shuffle: bool = False
    save_dataset: bool = True
    save_dataloader: bool = True
    filename: str = "./data/"
    num_workers: int = 0
    cutoff: float = 8.
    max_neighbors: int = 12
    keep_data_order: bool = False
    ratio_train: float = 0.8
    ratio_valid: float = 0.1
    ratio_test: float = 0.1

    # training configuration
    random_seed: Optional[int] = 123
    classification_threshold: Optional[float] = None
    target_multiplication_factor: Optional[float] = None
    epochs: int = 100
    metrics = F.mse_loss
    optimizer = "adamw"
    learning_rate = 1e-3
    weight_decay = 1e-05
    warmup_steps: int = 2000
    criterion: Literal["mse", "l1", "poisson", "zig"] = "mse"
    optimizer: Literal["adamw", "sgd"] = "adamw"
    scheduler: Literal["onecycle", "step", "none"] = "onecycle"
    standard_scalar_and_pca: bool = False
    distributed: bool = False
    n_early_stopping: Optional[int] = 50  # typically 50
    output_dir: str = os.path.abspath(".")

