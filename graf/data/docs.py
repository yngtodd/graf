import pickle
import shutil

import torch
import torch.nn as nn

from pathlib import Path
from torch_geometric.data import (
    Data, InMemoryDataset, DataLoader
)


class DocumentGraphs(InMemoryDataset):
    r"""Document graphs

    Every document is represented as its own
    graph, following Andrew's COO graph form.

    Args:
        root: root data path

        word_embedding: fixed word embeddings used as
                        features for every document graph

        transform: optional transform applied at load time

        pre_transform: optional transform when processesing
                       the raw data
    """

    raw_home = '/Users/ygx/src/python/trans/val_x.pkl'

    def __init__(self,
                 root,
                 word_embedding=None,
                 split="train",
                 transform=None,
                 pre_transform=None):

        self._check_split(split)
        self.word_embedding = word_embedding
        super(DocumentGraphs, self).__init__(root, transform, pre_transform)

    def _check_split(self, split):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(
                f"Data split must be either `train`, `valid`, or `test`!"
            )

        self.split = split

    @property
    def raw_file_names(self):
        return 'data.pkl'

    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt', 'test.pt']

    def download(self):
        shutil.copy(
            self.raw_home, Path(self.raw_dir).joinpath(self.raw_file_names)
        )

    def _load_edge_lists(self, path):
        """load the pickled data

        Args:
            path: path to graph pickle file

        Returns:
            edges: list of torch long tensors,
                   the edge list for each document graph
        """
        with open(path, 'rb') as f:
            edges = pickle.load(f)

        if self.split == "train":
            return edges
        else:
            raise ValueError('nope!')

    def process(self):
        torch.save(
            self.process_split('train'), self.processed_paths[0]
        )

    def process_split(self, split):
        # Pytorch geometric keeps this as a list of paths,
        # but we only have only raw graph pickle file
        raw_path = self.raw_paths[0]
        edges = self._load_edge_lists(raw_path)

        data_list = []
        for idx, edge_index in enumerate(edges):
            data = Data(
                x = None, #self.word_embedding,
                edge_index=edge_index,
                # number of unique nodes in the graph
                num_nodes=501 # vocab terms
            )

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data = self.collate(data_list)

        data[0].x = random_embeddings(501, 30)
        print(data[0])
        print(data[0].edge_index)
        print(data[0].x.shape)

        return data

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            Path(self.processed_dir).joinpath(f"data.pt")
        )

        return data


def random_embeddings(vocab_size, embed_dim):
    """Random word embeddings"""
    x = torch.arange(0, vocab_size)
    m = nn.Embedding(vocab_size, embed_dim)
    return m(x)


if __name__=='__main__':
    from torch_geometric.data import DenseDataLoader

    d = DocumentGraphs('/Users/ygx/data/docs')
    loader = DenseDataLoader(d, batch_size=32, shuffle=True)

    for data in loader:
        print(data)

