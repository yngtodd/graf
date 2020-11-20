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

    raw_home = "/Users/ygx/src/python/trans/val_x.pkl"

    def __init__(
        self,
        root,
        num_vocab,
        word_embedding=None,
        split="train",
        transform=None,
        pre_transform=None,
    ):

        self._check_split(split)
        self.num_vocab = num_vocab
        self.word_embedding = word_embedding
        super(DocumentGraphs, self).__init__(root, transform, pre_transform)

    def _check_split(self, split):
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Data split must be either `train`, `valid`, or `test`!"
            )

        self.split = split

    @property
    def raw_file_names(self):
        return ["val_x.pkl", "val_y.pkl"]

    @property
    def processed_file_names(self):
        return ["train.pt", "valid.pt", "test.pt"]

    def download(self):
        for filename in self.raw_file_names:
            shutil.copy(
                self.raw_home,
                Path(self.raw_dir).joinpath(filename)
            )

    def _load_edge_lists(self, path):
        """load the pickled data

        Args:
            path: path to graph pickle file

        Returns:
            edges: list of torch long tensors,
                   the edge list for each document graph
        """
        p = Path(path)

        with open(p.joinpath('val_x.pkl'), "rb") as f:
            edges = pickle.load(f)

        with open(p.joinpath('val_y.pkl'), "rb") as f:
            target = pickle.load(f)

        return edges, target

    def process(self):
        torch.save(
            self.process_split("train"), self.processed_paths[0]
        )

    def process_split(self, split):
        edges, target = self._load_edge_lists(self.raw_dir)

        x = random_embeddings(self.num_vocab, 30)

        data_list = []
        for doc in range(len(edges)):
            data = Data(
                x=x[doc],
                y=target[doc],
                edge_index=edges[doc],
                num_nodes=self.num_vocab
            )

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)

        return data

    def len(self):
        return 1

    def get(self, idx):
        data = torch.load(
            Path(self.processed_dir).joinpath(f"{self.split}.pt")
        )

        return data


def random_embeddings(vocab_size, embed_dim):
    """Random word embeddings"""
    x = torch.arange(0, vocab_size)
    m = nn.Embedding(vocab_size, embed_dim)
    return m(x)


if __name__ == "__main__":
    from torch_geometric.data import DenseDataLoader

    d = DocumentGraphs("/Users/ygx/data/docs", num_vocab=501)
    loader = DenseDataLoader(d, batch_size=32, shuffle=True)

    for data in loader:
        print(data)
