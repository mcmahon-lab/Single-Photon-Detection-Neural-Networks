import torch.nn.functional as F
import torch.nn as nn
from PhotonActivation import PhotonActivation, PhotonActivationCoh

class PDMLP_1(nn.Module):
    
    def __init__(self, n_hidden=100, n_input=784, n_output=10, dropout=None, last_layer_bias=False, sampler='bernoulli'):
        super(PDMLP_1, self).__init__()
        
        self.sampler = sampler
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(n_input, n_hidden, bias=False)
        self.act = PhotonActivation(sampler=sampler)
        self.fc2 = nn.Linear(n_hidden, n_output, bias=last_layer_bias)
        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, n_rep=1, slope=1.):
        x = x.view(-1, self.n_input)
        x_fc1 = self.act(self.fc1(x), n_rep=n_rep, slope=slope)
        if self.dropout is not None:
            x_fc1 = self.dropout(x_fc1)
        x_fc2 = self.fc2(x_fc1)
        x_out = F.log_softmax(x_fc2, dim=1)
        return x_out

class incoh_PDMLP(nn.Module):
    
    def __init__(self, n_hiddens=[100,100], n_input=784, n_output=10, sampler='bernoulli',output_bias=True):
        super(incoh_PDMLP, self).__init__()
        
        self.sampler = sampler
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_hiddens = n_hiddens
        
        n_nodes = [n_input]+list(n_hiddens)
        self.fcs = nn.ModuleList([nn.Linear(i,j,bias=False) for i, j in zip(n_nodes[:-1], n_nodes[1:])])
        self.last_fc = nn.Linear(n_hiddens[-1],n_output,bias=output_bias)
        self.act = PhotonActivation(sampler=sampler)

    def forward(self, x, n_rep=1, slope=1.):
        x = x.view(-1, self.n_input)
        for fc in self.fcs:
            x = fc(x)
            x = self.act(x, n_rep=n_rep, slope=slope)
        x_out = F.log_softmax(self.last_fc(x), dim=1)
        return x_out
    
class coh_PDMLP(nn.Module):
    
    def __init__(self, n_hiddens=[100,100], n_input=784, n_output=10, sampler='bernoulli',output_bias=True):
        super(coh_PDMLP, self).__init__()
        
        self.sampler = sampler
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_hiddens = n_hiddens
        
        n_nodes = [n_input]+list(n_hiddens)
        self.fcs = nn.ModuleList([nn.Linear(i,j,bias=False) for i, j in zip(n_nodes[:-1], n_nodes[1:])])
        self.last_fc = nn.Linear(n_hiddens[-1],n_output,bias=output_bias)
        self.act = PhotonActivationCoh(sampler=sampler)

    def forward(self, x, n_rep=1, slope=1.):
        x = x.view(-1, self.n_input)
        for fc in self.fcs:
            x = fc(x)
            x = self.act(x, n_rep=n_rep, slope=slope)
        x_out = F.log_softmax(self.last_fc(x), dim=1)
        return x_out