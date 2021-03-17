from typing import Any, Dict
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 1024
DEPTH = 2

class MLP(nn.Module):
    """Simple MLP suitable for recognizing single characters."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dim = np.prod(data_config["input_dims"])
        num_classes = len(data_config["mapping"])

        width = self.args.get("width", WIDTH)
        depth = self.args.get("numlayers", DEPTH)

        self.dropout = nn.Dropout(0.5)
        self.fcs = nn.ModuleList()
        for i in range(depth+1):
            l = input_dim if i == 0 else width
            r = num_classes if i == depth else width
            print(f"{i} <- Linear({l}, {r})")
            self.fcs.append(nn.Linear(l, r))

    def forward(self, x):
        x = torch.flatten(x, 1)
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
            if i == len(self.fcs) - 1:
                return x
            x = F.relu(x)
            x = self.dropout(x)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--numlayers", type=int, default=2)
        parser.add_argument("--width", type=int, default=1024)
        return parser
