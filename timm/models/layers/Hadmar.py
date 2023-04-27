import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import hadamard


class Hadmar(nn.Module):
    def __init__(self):
        super(Hadmar, self).__init__()

    def forward(self, x):

        # ----------------------------------
        x2, x3, x4 = x.size()
        a = hadamard(x3)
        b = hadamard(x4)
        self.h1 = torch.from_numpy(a.astype(np.float32)).cuda()
        self.h2 = torch.from_numpy(b.astype(np.float32)).cuda()
        x = torch.matmul(self.h1, x)

        x = torch.matmul(x, self.h2)
        # -----------------------------
        x = x/((x3*x4)**0.500)
        # --------------------------------

        return x

    # def extra_repr(self):
    #     return 'order={}, hardma={}'.format(
    #         self.order, self.hardma is not None
    #     )
