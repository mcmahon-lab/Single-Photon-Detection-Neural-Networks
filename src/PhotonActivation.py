import torch
import torch.nn as nn
from torch.autograd import Function
    
class PhotonCountingP(nn.Module):
    """ The probability of 1 photon in photon counting 
        (also the expectation value) with mean flux x """
    def __init__(self):
        super(PhotonCountingP, self).__init__()

    def forward(self, x):
        return 1.-torch.exp(torch.abs(x)*-1.)
    
class BernoulliFunctionST(Function):
    """ The 'Straight Through' stochastic Bernoulli activation"""
    @staticmethod
    def forward(ctx, input):

        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output

class PoissonFunctionST(Function):
    """ The 'Straight Through' stochastic Poisson activation"""
    @staticmethod
    def forward(ctx, input):

        return torch.poisson(input)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output
    
PoissonST = PoissonFunctionST.apply    
BernoulliST = BernoulliFunctionST.apply   

class PhotonActivation(nn.Module):

    def __init__(self,sampler='bernoulli'):
        super(PhotonActivation, self).__init__()
        self.act = PhotonCountingP()
        if sampler == 'poisson':
            self.sampler = PoissonST
        elif sampler == 'bernoulli':
            self.sampler = BernoulliST
        else:
            raise

    def forward(self, input, n_rep=1, slope=1.):
        x = input
        probs = self.act(slope * x)
        out = self.sampler(probs)
        if self.sampler == BernoulliST:
            probs = self.act(x)
        elif self.sampler == PoissonST:
            probs = torch.abs(x)
        else: raise
        if n_rep==0:  # Infinite number of shots per activation
            out = probs
        else:
            out = self.sampler(probs.unsqueeze(0).repeat((n_rep,)+(1,)*len(probs.shape))).mean(axis=0)*torch.sign(x)
        return out
        out = self.sampler(probs)
        return out

class PhotonActivationCoh(nn.Module):

    def __init__(self,sampler='bernoulli'):
        super(PhotonActivationCoh, self).__init__()
        self.act = PhotonCountingP()
        if sampler == 'poisson':
            self.sampler = PoissonST
        elif sampler == 'bernoulli':
            self.sampler = BernoulliST
        else:
            raise

    def forward(self, input, n_rep=1, slope=1.):
        x = input**2
        probs = self.act(slope * x)
        out = self.sampler(probs)
        if self.sampler == BernoulliST:
            probs = self.act(x)
        elif self.sampler == PoissonST:
            probs = torch.abs(x)
        else: raise
        if n_rep==0:  # Infinite number of shots per activation
            out = probs
        else:
            out = self.sampler(probs.unsqueeze(0).repeat((n_rep,)+(1,)*len(probs.shape))).mean(axis=0)*torch.sign(x)
        return out