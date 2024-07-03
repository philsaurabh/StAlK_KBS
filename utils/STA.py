import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from utils.memory import ContrastMemory
from utils.losses import SALoss

eps = 1e-7

class STA_Loss(nn.Module):
  
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(STA_Loss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_p, opt.nce_k, opt.nce_t, opt.nce_m)

        self.relation_loss = anchor_relation_loss(opt.nce_t)
        self.anchor_type = opt.anchor_type
        self.class_anchor = opt.class_anchor

    def forward(self, f_s, f_t, idx, batch_label, class_index, num_pos, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t) #after l2-norm, the norm of f_s and f_t is 1.

        batch_label = torch.argmax(batch_label, axis=1)
        batch_label_matrix = torch.eq(batch_label.view(-1, 1), batch_label.view(1, -1))

        out_s, out_t, self.memory_s, self.memory_t = self.contrast(f_s, f_t, idx, batch_label_matrix, contrast_idx)

        s_anchors, t_anchors = None, None
        for i in range(len(class_index)):
            if self.anchor_type == "center":
                img_index = torch.tensor(class_index[i]).cuda()

                s_anchors_i = torch.index_select(self.memory_s.cuda(), 0, img_index.view(-1))
                s_center_i = torch.mean(F.relu(s_anchors_i), axis=0, keepdims=True)
                s_anchors = s_center_i if s_anchors is None else torch.cat((s_anchors, s_center_i), axis=0)

                t_anchors_i = torch.index_select(self.memory_t.cuda(), 0, img_index.view(-1))
                t_center_i = torch.mean(F.relu(t_anchors_i), axis=0, keepdims=True)
                t_anchors = t_center_i if t_anchors is None else torch.cat((t_anchors, t_center_i), axis=0)

            elif self.anchor_type == "class":
                img_index = torch.tensor(np.random.permutation(class_index[i])[0:self.class_anchor]).cuda()
                s_anchors_i = torch.index_select(self.memory_s.cuda(), 0, img_index.view(-1))
                s_anchors = s_anchors_i if s_anchors is None else torch.cat((s_anchors, s_anchors_i), axis=0)

                t_anchors_i = torch.index_select(self.memory_t.cuda(), 0, img_index.view(-1))
                t_anchors = t_anchors_i if t_anchors is None else torch.cat((t_anchors, t_anchors_i), axis=0)

        relation_loss = self.relation_loss(f_s, s_anchors, f_t, t_anchors)
        return relation_loss
    
class anchor_relation_loss(nn.Module):
    """
    Compute centroid features of all classes.
    :param f: batch features. [bs, feat_dim]
    :param anchors: centroid features of seven classes. [n_anchors, feat_dim]
    :return: loss between the relation matrix of the student and ema teacher.
    """
    def __init__(self, T):
        super(anchor_relation_loss, self).__init__()
        self.l2norm = Normalize(2)
        self.T = T
        self.STA = SALoss().cuda()
        self.cos = nn.CosineEmbeddingLoss().cuda()
        self.target = torch.Tensor([1]).cuda() # for dissimilarity

    def forward(self, f_s, s_anchors, f_t, t_anchors):
        s_anchors = self.l2norm(s_anchors)
        s_relation = torch.div(torch.mm(f_s, s_anchors.clone().detach().T), self.T) # [bs, n_anchors]

        t_anchors = self.l2norm(t_anchors)
        t_relation = torch.div(torch.mm(f_t, t_anchors.clone().detach().T), self.T) # [bs, n_anchors]

        loss = self.STA(t_relation.detach(), s_relation)

        return loss

class Embed(nn.Module):
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x
    
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
