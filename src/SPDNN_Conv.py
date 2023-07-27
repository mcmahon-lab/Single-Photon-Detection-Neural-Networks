import torch.nn.functional as F
import torch.nn as nn
from PhotonActivation import PhotonActivationCoh
import numpy as np
    
def PDConv(n_in=128, n_out=128, s=1, ks=3, batchnorm=True):
    if batchnorm:
        return [
            nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=ks, stride=s, padding=int((ks-1)/2*s),bias=False),
            nn.BatchNorm2d(n_out),
            PhotonActivationCoh()
        ]
    else:
        return [
            nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=ks, stride=s, padding=int((ks-1)/2*s),bias=False),
            PhotonActivationCoh()
        ]
    
def PDConvsAP(n_in=3, n_chan=[128,128], ss=[1,1], kss=[3,3], batchnorm=True):
    modules = []
    n_list = [n_in]+list(n_chan)
    for i in range(len(n_chan)):
        modules += PDConv(n_in=n_list[i],n_out=n_list[i+1],s=ss[i],ks=kss[i],batchnorm=batchnorm)\
                      +[nn.AvgPool2d((2,2))]

    return nn.Sequential(*modules)

class PDConvNet(nn.Module):
    
    def __init__(self, n_linear=100, n_output=10, d_input=(1,28,28), n_chan=[128,128], ss=[1,1], kss=[3,3], batchnorm=True, dropout=None, last_layer_bias=True, linear_act='PD', sampler='bernoulli'):
        super(PDConvNet, self).__init__()
        
        self.sampler = sampler
        self.n_chan = n_chan
        self.batchnorm = batchnorm
        
        self.d_input = d_input
        self.n_output = n_output
        self.n_linear = n_linear

        self.convs = PDConvsAP(n_in=d_input[0],n_chan=n_chan,ss=ss,kss=kss,batchnorm=batchnorm)
        self.flat = nn.Flatten()
        
        self.fc1 = nn.Linear(int(n_chan[-1]*(d_input[1]//2**len(n_chan)//np.prod(ss))**2), n_linear, bias=False)
        if linear_act=='PD':
            self.linear_act = PhotonActivationCoh(sampler=sampler)
        elif linear_act=='ReLU':
            self.linear_act = nn.ReLU()
        self.fc2 = nn.Linear(n_linear, n_output, bias=last_layer_bias)
        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, n_rep=1, slope=1.):
        x = x.view(-1, *self.d_input)
        for layer in self.convs:
            if isinstance(layer, PhotonActivationCoh):
                x = layer(x, n_rep=n_rep, slope=slope)
            else:
                x = layer(x)
        x = self.flat(x)
        x = self.fc1(x)
        if isinstance(self.linear_act, PhotonActivationCoh):
            x = self.linear_act(x, n_rep=n_rep, slope=slope)
        else:
            x = self.linear_act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        x_out = F.log_softmax(x, dim=1)
        return x_out