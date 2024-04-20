import math
import torch
import torch.nn as nn




class _Identity(nn.Module):
    def forward(self,x):
        return x



class Resnet20(nn.Module):
    def __init__(self,block,quan_first=False,quan_last=False,constr_activation=None):
        super(Resnet20,self).__init__()
        self.quan_first=quan_first
        self.quan_last=quan_last
        self.quan_activation=constr_activation is not None

        if quan_first:
            self.first_act=nn.ReLU(constr_activation) if self.quan_activation else _Identity()
            self.conv1=Conv2d
