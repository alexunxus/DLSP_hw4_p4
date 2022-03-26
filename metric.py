import torch

def acc(pred, gt):
    '''
        pred: should be [B,]
        gt  : should be [B,]
    '''
    return torch.sum(pred == gt)/pred.shape[0]