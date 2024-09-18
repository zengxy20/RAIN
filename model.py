import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from sklearn.metrics import roc_auc_score,roc_curve
import time
import random
from tqdm import *
torch.manual_seed(100)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.to(device)
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class ItemConv(Module):
    def __init__(self, layers, emb_size=100):
        super(ItemConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.w_item = {}
        for i in range(self.layers):
            self.w_item['weight_item%d' % (i)] = nn.Linear(self.emb_size, self.emb_size, bias=False)

    def forward(self, adjacency, embedding):
        adjacency = adjacency.multiply(1.0 / adjacency.sum(axis=0).reshape(1, -1))
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        ad = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        ad = trans_to_cuda(ad)
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = trans_to_cuda(self.w_item['weight_item%d' % (i)])(item_embeddings)
            item_embeddings = torch.sparse.mm(ad, item_embeddings)
            final.append(F.normalize(item_embeddings, dim=-1, p=2))
        item_embeddings = torch.sum(torch.stack(final).squeeze(),dim=0)/(self.layers+1)
        return item_embeddings

class RAIN(Module):
    def __init__(self, adjacency, adjacency_a, adjacency_1, ad1r, ad1c, n_node, lr, layers, lambda1, lambda2, dataset, emb_size=100, batch_size=100):
        super(RAIN, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.dataset = dataset
        self.ad1r = ad1r
        self.ad1c = ad1c
        self.lr = lr
        self.layers = layers
        self.lambda1 = lambda1
        self.lambda2 = lambda2 #[0.0001,10,1000,0.01,1]
        self.w_k = 10
        self.adjacency = adjacency
        self.adjacency_a = adjacency_a
        self.adjacency_1 = adjacency_1
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.ItemGraph = ItemConv(self.layers)
        self.pos_len = 300
        self.pos_embedding = nn.Embedding(self.pos_len, self.emb_size)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.w_3 = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        self.w_4 = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.mlp =  nn.Linear(100, 1)
        self.mlp1 = nn.Sequential(nn.Linear(100, 100), nn.Linear(100, 100))
        self.loss_recons = nn.MSELoss()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask,o):
        zeros = trans_to_cuda(torch.FloatTensor(1, self.emb_size)).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = trans_to_cuda(torch.FloatTensor(list(reversed_sess_item.shape)[0], list(reversed_sess_item.shape)[1], self.emb_size)).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(o, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        lenth = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:lenth]
        pos_emb = pos_emb.unsqueeze(0).repeat(list(reversed_sess_item.shape)[0], 1, 1)

        hs1 = hs.unsqueeze(-2).repeat(1, lenth, 1)
        hs2 = torch.div(torch.sum(seq_h, 1), session_len).unsqueeze(-2).repeat(1, lenth-1, 1)
        
        nh = torch.matmul(torch.cat([pos_emb, o], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh)+self.glu2(hs1))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        glo = torch.sum(beta * seq_h, 1)
        gate = torch.sigmoid(torch.matmul(glo,self.w_3) + torch.matmul(seq_h[:,0,:],self.w_4))
        select = (1 - gate) * seq_h[:,0,:] + gate * glo
        return select

    def forward(self, session_item, session_len, reversed_sess_item, mask, epoch, tar, train):
        item_embeddings_a = self.ItemGraph(self.adjacency_a, self.embedding.weight)

        ij = item_embeddings_a[self.ad1r] - item_embeddings_a[self.ad1c]
        pos_score = torch.sigmoid(self.mlp(ij))
        pos_score_1 = torch.sum(torch.exp(pos_score / 0.2), 1)
        j = np.random.randint(self.n_node, size=len(self.ad1c))
        ij = item_embeddings_a[self.ad1r] - item_embeddings_a[j]
        neg_score = torch.sigmoid(self.mlp(ij))
        neg_score_1 = torch.sum(torch.exp(neg_score / 0.2), 1)
        KD_loss1 = -torch.sum(torch.log(pos_score_1 / (pos_score_1 + neg_score_1)))

        lab = [1] * len(self.ad1r) + [0] * len(self.ad1c)
        pred = torch.cat([pos_score, neg_score], 0).squeeze().cpu().detach().numpy()
        cur_acc = roc_auc_score(lab, pred)
        ij_a = item_embeddings_a[self.ad1r] - item_embeddings_a[self.ad1c]
        score = torch.sigmoid(self.mlp(ij_a).squeeze())
        ones = trans_to_cuda(torch.ones(len(self.ad1r)))
        KD_loss2 = self.loss_recons(ones, score)

        ij_a = item_embeddings_a[self.adjacency_1.row] - item_embeddings_a[self.adjacency_1.col]
        score = torch.sigmoid(self.mlp(ij_a).squeeze())
        I = torch.bernoulli(1 / (1 + torch.exp((-torch.log(score) + 0.01)) / 1))
        index = torch.where(I == 1)
        if index[0].size()[0] > 0:
            ones = trans_to_cuda(torch.ones(len(index[0])))
            KD_loss3 = self.loss_recons(ones, score[index[0]])
        else:
            KD_loss3 = 0
        KD_loss2+=KD_loss3

        zeros = trans_to_cuda(torch.FloatTensor(1, self.emb_size)).fill_(0)
        item_embedding = torch.cat([zeros, self.embedding.weight], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = trans_to_cuda(
            torch.FloatTensor(list(reversed_sess_item.shape)[0], list(reversed_sess_item.shape)[1], self.emb_size)).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        lenth = seq_h.shape[1]
        ij_a = seq_h[:, 1:, :] - seq_h[:, :-1, :]
        score1 = torch.sigmoid(self.mlp(ij_a))
        o1 = torch.matmul(torch.softmax(torch.matmul(seq_h, torch.transpose(seq_h[:, 1:, :], 1, 2)) / 10, -1),score1.repeat(1,1,self.emb_size)).squeeze()
        ij_a = seq_h[:, :-1, :] - seq_h[:, 1:, :]
        score2 = torch.sigmoid(self.mlp(ij_a))
        o2 = torch.matmul(torch.softmax(torch.matmul(seq_h, torch.transpose(seq_h[:, :-1, :], 1, 2)) / 10, -1),score2.repeat(1,1,self.emb_size)).squeeze()
        o = o1+o2

        sess_emb_i_initial = self.generate_sess_emb(item_embeddings_a, session_item, session_len, reversed_sess_item,mask,o)
        sess_emb_i = self.w_k * F.normalize(sess_emb_i_initial, dim=-1, p=2)
        item_embeddings_i = F.normalize(item_embeddings_a, dim=-1, p=2)
        scores_item = torch.mm(sess_emb_i, torch.transpose(item_embeddings_i, 1, 0))
        loss_item = self.loss_function(scores_item, tar)

        te = trans_to_cuda(torch.tensor( [14596] ).long())
        item_embed = item_embedding[te]

        return loss_item, scores_item, self.lambda1*KD_loss1+self.lambda2*KD_loss2, cur_acc

def forward(model, i, data, epoch, train):
    tar, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar.astype(float)).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    loss_item, scores_item, KD_loss, cur_acc = model(session_item, session_len, reversed_sess_item, mask, epoch,tar, train)
    return tar, scores_item,  loss_item, KD_loss, cur_acc

def train_test(model, train_data, test_data, epoch):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    #slices = slices[:10]
    for i in tqdm(slices):
        model.zero_grad()
        tar, scores_item, loss_item, KD_loss, cur_acc = forward(model, i, train_data, epoch, train=True)
        loss = loss_item + KD_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [1, 5, 10, 20]
    metricss = {}
    for K in top_K:
        metricss['hit%d' % K] = []
        metricss['mrr%d' % K] = []
    metricss['AUC'] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    #slices = slices[:5]
    for i in slices:
        tar, scores_item, loss_item, KD_loss, cur_acc = forward(model, i, test_data, epoch, train=False)
        scores = scores_item.topk(20)[1]
        scores = trans_to_cpu(scores).detach().numpy()
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(scores[:, :K], tar):
                metricss['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metricss['mrr%d' % K].append(0)
                else:
                    metricss['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
        metricss['AUC'].append(cur_acc)
    return metricss, total_loss


