import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul
import numpy as np
from torch import autograd
from utils import *


class DTPNet(nn.Module):
    def __init__(self, opt, device):
        super(DTPNet, self).__init__()
        network_size = [784] + [240] * 7 + [10]
        feedforward = []
        self.lr_target = opt.lr_target
        self.C = opt.C
        for ind in range(len(network_size) - 1):
            feedforward.append(nn.Linear(network_size[ind], network_size[ind + 1]))
        self.ff = nn.ModuleList(feedforward)
        self.bs = opt.batch_size
        feedback = []
        for ind in range(1, len(network_size) - 2):
            feedback.append(nn.Linear(network_size[ind + 1], network_size[ind]))
        self.fb = nn.ModuleList(feedback)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                out_, in_ = m.weight.shape
                m.weight.data = torch.Tensor(rand_ortho((out_, in_), np.sqrt(6. / (out_ + in_))))
                m.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.shape[0], 784)
        ff_value = [x]
        for ind, m in enumerate(self.ff):
            if ind == len(self.ff) - 1:
                x = torch.softmax(m(x), dim=-1)
            else:
                x = torch.tanh(m(x))
            ff_value.append(x)
        return ff_value

    def inject_noise(self, forward_value):
        hc_values = []
        fhc_values = []
        for i in range(len(self.ff) - 3, -1, -1):
            h_c = gaussian(forward_value[i + 1], self.C)
            fh_c = torch.tanh(self.ff[i + 1](h_c))
            hc_values.append(h_c)
            fhc_values.append(fh_c)
        return hc_values, fhc_values

    def feedback(self, ff_value, y_label):
        fb_value = []
        cost = nll_loss(ff_value[-1], y_label)
        h_ = ff_value[-2] - self.lr_target * torch.autograd.grad(cost, ff_value[-2], retain_graph=True)[0]
        fb_value.append(h_)
        for i in range(len(self.fb) - 1, -1, -1):
            h = ff_value[i + 1]
            gh = torch.tanh(self.fb[i](ff_value[i + 2]))
            gh_ = torch.tanh(self.fb[i](fb_value[-1]))
            h_ = h - gh + gh_
            fb_value.append(h_)
        return fb_value, cost

    def set_gradient(self, x, y):
        ff_value = self.forward(x)
        fb_value, cost = self.feedback(ff_value, y)
        hc_value, fhc_value = self.inject_noise(ff_value)
        len_fb = len(self.fb)
        for idx, layer in enumerate(self.fb):
            in1 = fhc_value[len_fb - 1 - idx]
            in2 = hc_value[len_fb - 1 - idx]
            loss_local = mse(torch.tanh(layer(in1.detach())), in2.detach())
            layer.weight.grad, layer.bias.grad = autograd.grad(loss_local, layer.parameters())
        len_ff = len(self.ff)
        for idx, layer in enumerate(self.ff):
            if idx == len_ff - 1:
                layer.weight.grad, layer.bias.grad = autograd.grad(cost, layer.parameters())
            else:
                in1 = ff_value[idx]
                in2 = fb_value[len(fb_value) - 1 - idx]
                loss_local = mse(torch.tanh(layer(in1.detach())), in2.detach())
                layer.weight.grad, layer.bias.grad = autograd.grad(loss_local, layer.parameters())
        return ff_value, cost

    def forward_parameters(self):
        res = []
        for layer in self.ff:
            res += layer.parameters()
        return res

    def feedback_parameters(self):
        res = []
        for layer in self.fb:
            res += layer.parameters()
        return res
