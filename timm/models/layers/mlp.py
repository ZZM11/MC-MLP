""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from curses import noraw
from tkinter import Y
from torch import nn as nn
import torch

from .helpers import to_2tuple
from .Hadmar import Hadmar
from .dct import dct_2d, idct_2d
from functools import partial


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Mlp_dctmlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features*2, out_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        # self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        # self.drop2 = nn.Dropout(drop_probs[1])
        self.encode = dct_2d
        self.norm = nn.LayerNorm(in_features*2)

    def forward(self, x):
        y = self.encode(x)
        # print(y.shape)
        x = torch.cat((x, y), 2)
        # print()
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        # x = self.fc2(x)
        # x = self.drop2(x)
        return x


class Mlp_hadamamlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features*2, out_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        # self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        # self.drop2 = nn.Dropout(drop_probs[1])
        # self.encode = dct_2d
        self.hadama = Hadmar()
        self.norm = nn.LayerNorm(in_features*2)

    def forward(self, x):
        y = self.hadama(x)
        x = torch.cat((x, y), 2)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        # x = self.fc2(x)
        # x = self.drop2(x)
        return x

class Mlp_dctfc(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        # self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.drop2 = nn.Dropout(drop_probs[0])
        self.fc1 = nn.Linear(in_features*3, hidden_features, bias=bias[1])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        # self.drop2 = nn.Dropout(drop_probs[1])
        self.hadama = Hadmar()
        self.encode = dct_2d
        self.norm1 = norm_layer(in_features*3)
        # self.norm2 = norm_layer(hidden_features*2)

    def forward(self, x):
        # print(x.shape)
        y1 = self.encode(x)
        y2 = self.hadama(x)
        y = torch.cat((x, y1, y2), 2)
        # print(y.shape)
        # y = self.encode(y1)
        # y = torch.cat((y1, y), 2)
        # x = self.fc1(y)
        # x = self.drop1(x)
        y = self.norm1(y)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
# ----------------------------------------------
        # x = self.hadama(y)
        # y = torch.cat((x, y), 2)
        # print(y.shape)
        # y = self.encode(y1)
        # y = torch.cat((y1, y), 2)
        # x = self.fc1(y)
        # x = self.drop1(x)
        # y = self.norm2(y)
        y = self.fc2(y)
        # y = self.act(y)
        y = self.drop2(y)
        return y


class Mlp_dct(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features//2, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        # self.hadama = Hadmar()
        # self.encode = dct_2d

    def forward(self, x): 
        x = self.fc1(x)
        # print(x.shape)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        # print(x.shape)
        x = self.drop2(x)
        return x


class Mlp_cat_nofc1(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(in_features, out_features//4)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.act(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop2(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
            gate_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            # FIXME base reduction on gate property?
            hidden_features = hidden_features // 2
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp_dctfc(nn.Module):
    """ MLP as used in gMLP
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
            gate_layer=None, bias=True, drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features*2, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            # FIXME base reduction on gate property?
            hidden_features = hidden_features // 2
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        # self.hadama = Hadmar()
        self.encode = dct_2d
        self.norm = norm_layer(in_features*2)

    def forward(self, x):
        # print(x.shape)
        y = self.encode(x)
        x = torch.cat((x, y), 2)
        # print(x.shape)
        # y = self.encode(x)
        # x = torch.cat((x, y), 2)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop2(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features,
                             kernel_size=1, bias=bias[0])
        self.norm = norm_layer(
            hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features,
                             kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
