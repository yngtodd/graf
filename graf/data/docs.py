import torch
import pickle

from pathlib import Path
from torch_geometric.data import (
    Data, Dataset
)


class DocumentGraphs(Dataset):
    r"""Document graphs

    Every document is represented as its own
    graph, following Andrew's COO graph form.

    Args:
        root: root data path
        transform: optional transform applied at load time
        pre_transform: optional transform when processesing 
                       the raw data
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(DocumentGraphs, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ["graph.-1_500.pkl"]

    @property
    def processed_file_names(self):
        path = Path(self.processed_dir).glob("*.pt")
        return [f.name for f in path]

    def _load_edge_lists(self, path):
        """load the pickled edge lists

        Args:
            path: path to graph pickle file
        
        Returns:
            edges: list of torch long tensors,
                   the edge list for each document graph 
        """
        with open(path, 'rb') as f:
            edges = pickle.load(f)

        return edges

    def process(self):
        # Pytorch geometric keeps this as a list of paths,
        # but we only have only raw graph pickle file
        raw_path = self.raw_paths[0]
        edges = self._load_edge_lists(raw_path)

        for idx, edge_index in enumerate(edges):
            data = Data(edge_index=edge_index)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(
                data, Path(self.processed_dir).joinpath(f"data_{idx}.pt")
            )


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            Path(self.processed_dir).joinpath(f"data_{idx}.pt")
        )

        return data

