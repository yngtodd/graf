import torch.nn as nn

from typing import Dict


class MultitaskClassifier(nn.Module):
    """ Multi-task Classifier

    Args:
        input_dim: input dimension for each of the linear layers
        tasks: dictionary of tasks and their respective number of classes
    """

    def __init__(self, input_dim: int, tasks: Dict[str, int]):
        super(MultitaskClassifier, self).__init__()
        self.tasks = tasks

        for task, num_classes in tasks.items():
            self.add_module(
                task,
                nn.Linear(input_dim, num_classes)
            )

    def num_classes(self, task):
        """ Get number of classes for a task. """
        return self.tasks[task]

    def forward(self, x):
        logits = {}
        for task, _ in self.tasks.items():
            logits[task] = self._modules[task](x)

        return logits
