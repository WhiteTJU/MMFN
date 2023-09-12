import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from model.rvs import rvs
import pdb
# import model.resnet as resnet
import resnet as resnet
from torch.autograd import Variable
from torchstat import stat
import argparse
from torch import Tensor

# from .utils import load_state_dict_from_url
# from typing import Type, Any, Callable, Union, List, Optional

pretrained_path = "/home/zdd/sa_mvcnn/ta34/experiment/pre/resnet34-333f7ec4.pth"
pretrained_path1 = "/home/zdd/sa_mvcnn/ta34/experiment/pre/pixelShuffle_rotation_epoch_31_acc_0.9363.pth"


class TA34(nn.Module):
    def __init__(self, args, num_classes=40, pretrained=True):
        super(TA34, self).__init__()

        self.pretrained = pretrained
        self.args = args
        # self.Bk5 = Block5(args, num_classes)
        self.Bk5 = Block9(args, num_classes)

        # -------------------------------------------------------------------------

        # self.TA1 = TANet_3(args, self.args.views, 64, 56, state=1)
        # self.TA2 = TANet_3(args, self.args.views, 128, 28, state=1)
        # self.TA3 = TANet_3(args, self.args.views, 256, 14, state=2)
        # self.TA4 = TANet_3(args, self.args.views, 512, 7, state=2)

        # -------------------------------------------------------------------------

        # self.TA1 = TANet_3(args, self.args.views, 64, 56, state=1)
        # self.TA2 = TANet_3(args, self.args.views, 128, 28, state=1)
        # self.TA11 = TANet_3(args, self.args.views, 64, 56, state=2)
        # self.TA22 = TANet_3(args, self.args.views, 128, 28, state=2)

        # -------------------------------------------------------------------------
        self.Bk1 = resnet.__dict__['resnet34_1']()
        self.Bk2 = resnet.__dict__['resnet34_2']()
        self.Bk3 = resnet.__dict__['resnet34_3']()
        self.Bk4 = resnet.__dict__['resnet34_4']()
        state_dict = torch.load(pretrained_path, map_location='cuda:' + self.args.gpu)
        state_dict1 = {}
        state_dict2 = {}
        state_dict3 = {}
        state_dict4 = {}

        j = []
        for i in state_dict:
            j.append(i)
        k = 0
        for i in j:
            k = k + 1
            if (0 < k < 36):
                state_dict1[i] = state_dict.pop(i)
            if (35 < k < 81):
                state_dict2[i] = state_dict.pop(i)
            if (80 < k < 146):
                state_dict3[i] = state_dict.pop(i)
            if (145 < k < 181):
                state_dict4[i] = state_dict.pop(i)

        self.Bk1.load_state_dict(state_dict1)
        self.Bk2.load_state_dict(state_dict2)
        self.Bk3.load_state_dict(state_dict3)
        self.Bk4.load_state_dict(state_dict4)
        print("Use Pre-Bknet")

        # -------------------------------------------------------------------------
        # self.Bk1 = resnet.__dict__['resnet34_1']()
        # self.Bk2 = resnet.__dict__['resnet34_2']()
        # self.Bk3 = resnet.__dict__['resnet34_3']()
        # self.Bk4 = resnet.__dict__['resnet34_4']()
        # state_dict = torch.load(pretrained_path1, map_location='cuda:' + self.args.gpu)
        # state_dict1 = {}
        # state_dict2 = {}
        # state_dict3 = {}
        # state_dict4 = {}
        # state_dict5 = {}
        # state_dict6 = {}
        # state_dict7 = {}
        # state_dict8 = {}
        #
        # j = []
        # for i in state_dict:
        #     j.append(i)
        # for i in j:
        #     if (i.find("Bk1.") == 0):
        #         state_dict1[i[4:]] = state_dict.pop(i)
        #     if (i.find("Bk2.") == 0):
        #         state_dict2[i[4:]] = state_dict.pop(i)
        #     if (i.find("Bk3.") == 0):
        #         state_dict3[i[4:]] = state_dict.pop(i)
        #     if (i.find("Bk4.") == 0):
        #         state_dict4[i[4:]] = state_dict.pop(i)
        #     if (i.find("Bk5.") == 0):
        #         state_dict5[i[4:]] = state_dict.pop(i)
        #     if (i.find("TA1.") == 0):
        #         state_dict6[i[4:]] = state_dict.pop(i)
        #     if (i.find("TA2.") == 0):
        #         state_dict7[i[4:]] = state_dict.pop(i)
        #     if (i.find("TA3.") == 0):
        #         state_dict8[i[4:]] = state_dict.pop(i)
        #
        # self.Bk1.load_state_dict(state_dict1)
        # self.Bk2.load_state_dict(state_dict2)
        # self.Bk3.load_state_dict(state_dict3)
        # self.Bk4.load_state_dict(state_dict4)
        # # self.Bk5.load_state_dict(state_dict5)
        # self.TA1.load_state_dict(state_dict6)
        # self.TA2.load_state_dict(state_dict7)
        # self.TA3.load_state_dict(state_dict8)
        # print("Use Pre-Bknet")

        # -------------------------------------------------------------------------
        # self.conv = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )
        # -------------------------------------------------------------------------


    def forward(self, x):
        x = x.reshape(-1, 3, 224, 224)
        x = self.Bk1(x)
        # x = self.TA1(x) + x
        x = self.Bk2(x)
        # x = self.TA2(x) + x
        x = self.Bk3(x)
        # x = self.TA3(x) + x
        x = self.Bk4(x)
        x, fts = self.Bk5(x)


        return x, fts

        # -------------------------------------------------------------------------


class TANet_3(nn.Module):
    def __init__(self, args, V, C, H, state):
        super(TANet_3, self).__init__()
        self.args = args
        self.state = state

        if state == 1:
            self.att11 = Attention(V * C, H * H)

        if state == 2:
            self.att12 = Attention(V * H * H, C)

        if state == 3:
            self.att11 = Attention(V * C, H * H)
            self.att12 = Attention(V * H * H, C)

    def forward(self, x):
        # x:    (B*V,196,14,14)
        # feat: (B,V,196,1)
        V = int(self.args.views)
        B = int(x.shape[0] / V)
        C = int(x.shape[1])
        H = int(x.shape[2])

        x = x.reshape(B, V, C, H * H)

        if self.state == 1:
            x1 = x.transpose(1, 3).transpose(2, 3).reshape(B, H * H, V * C)
            x1 = self.att11(x1)
            x1 = x1.reshape(B, 1, 1, H * H).repeat(1, V, C, 1)
            x = x * x1

        if self.state == 2:
            x2 = x.transpose(1, 2).reshape(B, C, V * H * H)
            x2 = self.att12(x2)
            x2 = x2.reshape(B, 1, C, 1).repeat(1, V, 1, H * H)
            x = x * x2

        x = x.reshape(B * V, C, H, H)

        return x


class customizedModule(nn.Module):
    def __init__(self):
        super(customizedModule, self).__init__()

    # linear transformation (w/ initialization) + activation + dropout
    def customizedLinear(self, in_dim, out_dim, activation=None, dropout=False):
        cl = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.xavier_uniform(cl[0].weight)
        nn.init.constant(cl[0].bias, 0)

        if activation is not None:
            cl.add_module(str(len(cl)), activation)
        if dropout:
            cl.add_module(str(len(cl)), nn.Dropout(p=self.args.dropout))

        return cl


class Attention(customizedModule):
    def __init__(self, hidden_dim, views):  # hidden_dim=512, views=12
        super().__init__()
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(hidden_dim, 128)
        self.w15 = nn.Linear(128, 16)
        self.w2 = nn.Linear(16, 1)

    def forward(self, x):
        weight = F.softmax(x, dim=-2)
        # weight = x
        weight = weight.squeeze(0)
        weight = self.w1(weight)
        weight = self.w15(weight)
        weight = self.w2(weight)
        weight = weight.unsqueeze(0)

        return weight

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # if mask is not None:
        #     attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv

        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # q = q.unsqueeze(0)
        # k = k.unsqueeze(0)
        # v = v.unsqueeze(0)
        # print(q.shape)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # if mask is not None:
        #     mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = q.squeeze(0)
        q = self.dropout(self.fc(q))
        q = q.unsqueeze(0)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class Block9(nn.Module):
    def __init__(self, args, num_classes=40):
        super(Block9, self).__init__()

        self.args = args

        self.dim_reduction_conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.av1pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes),
        )

        self.fcf = nn.Linear(512, 512)

        self.fcb1 = nn.Linear(512, 512)
        self.fcb2 = nn.Linear(512, 256)

        # self.slf_attn = MultiHeadAttention(3, 512, 512, 512, dropout=0.1)



    def forward(self, x):
        x = self.dim_reduction_conv2(x)
        x = self.av1pool(x)
        x = x.reshape(-1, 512)
        x = self.fcf(x)

        x = x.reshape(-1, self.args.views, 512)
        # --------------------------------------------------------
        # x, _ = self.slf_attn(x, x, x)
        x = torch.sum(x, dim=1)
        x = self.fcb1(x)
        x = self.fcb2(x)
        fts = x

        x = self.classifier(fts)

        return x, fts

if __name__ == '__main__':
    def parse_args():
        '''PARAMETERS'''
        parser = argparse.ArgumentParser('PointNet')
        parser.add_argument('--batchsize', type=int, default=1, help='batch size in training')
        parser.add_argument('--epoch', default=80, type=int, help='number of epoch in training')
        parser.add_argument('--j', default=4, type=int, help='number of epoch in training')
        parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
        parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training SGD or Adam')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
        parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                            help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        parser.add_argument('--wd', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--stage', type=str, default='train', help='train test, extract feature')
        parser.add_argument('--views', default=20, type=int, help='the number of views')
        parser.add_argument('--num_classes', default=40, type=int, help='the number of clsses')
        parser.add_argument('--model_name', type=str, default='pixelShuffle_rotation', help='train test')
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--c', default=5.0, type=float)
        parser.add_argument('--r', default=4, type=int)
        parser.add_argument('--word_dim', default=512, type=int)
        return parser.parse_args()
    args = parse_args()

    model = TA34(args)
    stat(model, (60, 224, 224))