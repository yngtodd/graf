import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self, num_vocab, embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(num_vocab, embed_dim)

    def forward(self, x):
        self.embed(x)
