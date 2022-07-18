import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(DEVICE)
    return nn.BCELoss()(ad_out, dc_target)

def cosine_matrix(x,y):
    x=F.normalize(x,dim=1)
    y=F.normalize(y,dim=1)
    xty=torch.sum(x.unsqueeze(1)*y.unsqueeze(0),2)
    return 1-xty


def SM(Xs, Xt, Ys, Yt, Cs_memory, Ct_memory, Wt=None, decay=0.3):
    Cs = Cs_memory.clone()
    Ct = Ct_memory.clone()
    K = Cs.size(0)
    for k in range(K):
        Xs_k = Xs[Ys == k]
        Xt_k = Xt[Yt == k]
        if len(Xs_k) == 0:
            Cs_k = 0.0
        else:
            Cs_k = torch.mean(Xs_k, dim=0)

        if len(Xt_k) == 0:
            Ct_k = 0.0
        else:
            if Wt is None:
                Ct_k = torch.mean(Xt_k, dim=0)
            else:
                Wt_k = Wt[Yt == k]
                Ct_k = torch.sum(Wt_k.view(-1, 1) * Xt_k, dim=0) / (torch.sum(Wt_k) + 1e-5)
        Cs[k, :] = (1 - decay) * Cs_memory[k, :] + decay * Cs_k
        Ct[k, :] = (1 - decay) * Ct_memory[k, :] + decay * Ct_k
    Dist = cosine_matrix(Cs, Ct)
    return torch.sum(torch.diag(Dist)), Cs, Ct


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_( 1, -2, inputs, inputs.t())
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        p_dist = []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().view(1))
            dist_an.append(dist[i][mask[i] == 0].min().view(1))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        prec = (dist_an.data > dist_ap.data).float().mean()
        ########
        # Compute ranking hinge loss
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # prec = Variable(torch.from_numpy(prec).long().cuda())
        return loss, prec
