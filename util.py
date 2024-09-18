import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from operator import itemgetter
import random
np.random.seed(100)
from tqdm import *
import torch
def data_masks(all_sessions, n_node,gamma):
    adj = dict()
    numdrop = 1
    for sess in all_sessions:
        for i, item in enumerate(sess):
            if i == len(sess)-1:
                break
            else:
                if sess[i] - 1 not in adj.keys():
                    adj[sess[i]-1] = dict()
                    adj[sess[i]-1][sess[i]-1] = 1
                    adj[sess[i]-1][sess[i+1]-1] = 1
                else:
                    if sess[i+1]-1 not in adj[sess[i]-1].keys():
                        adj[sess[i] - 1][sess[i + 1] - 1] = 1
                    else:
                        adj[sess[i]-1][sess[i+1]-1] += 1
    row, col, data = [], [], []
    ad1_row,ad1_col = [],[]
    num = 0
    num_neigh = torch.ones(n_node)
    for i in adj.keys():
        item = adj[i]
        num_neigh[i] = len(item)
        for j in item.keys():
            row.append(i)
            col.append(j)
            data.append(adj[i][j])
    coo = coo_matrix((data, (row, col)), shape=(n_node, n_node))

    sorted_id = sorted(range(len(data)), key=lambda k: data[k], reverse=True)
    confident_id = sorted_id[:int(gamma*len(sorted_id))]
    unconfident_id = sorted_id[int(gamma*len(sorted_id)):]
    np.random.seed(2023)
    mask = np.random.randint(len(confident_id),size=int(len(confident_id)*0.4))
    mask_row, mask_col = [], []
    remain_index =[]
    for i, j in enumerate(confident_id):
        if i not in mask:
            remain_index.append(j)
        else:
            mask_row.append(row[j])
            mask_col.append(col[j])
    n_row, n_col, n_data = [], [], []
    for i in remain_index:
        n_row.append(row[i])
        n_col.append(col[i])
        n_data.append(data[i])
    ad1_row, ad1_col, ad1_data = [], [], []
    for i in unconfident_id:
        ad1_row.append(row[i])
        ad1_col.append(col[i])
        ad1_data.append(data[i])
    coo_a = coo_matrix((n_data, (n_row, n_col)), shape=(n_node, n_node))
    coo_1 = coo_matrix((ad1_data, (ad1_row, ad1_col)), shape=(n_node, n_node))
    return coo, coo_a,coo_1,mask_row,mask_col,num_neigh

class Data():
    def __init__(self, data, all_train, gamma,shuffle=False, n_node=None):
        self.raw = np.asarray(data[0], dtype=object)
        adj,adj_a,adj_1,ad1_row,ad1_col,num_neigh = data_masks(all_train, n_node, gamma)
        # # print(adj.sum(axis=0))
        self.adjacency = adj
        self.adjacency_a = adj_a
        self.adjacency_1 = adj_1
        self.ad1r = ad1_row
        self.ad1c = ad1_col
        self.num_neigh = num_neigh
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        # matrix = self.dropout(matrix, 0.2)
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)  # 随机打乱顺序
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        if n_batch > 1:
            slices = np.split(np.arange(n_batch * batch_size), n_batch)
            slices[-1] = np.arange(self.length - batch_size, self.length)
        else:
            slices = np.split(np.arange(n_batch * batch_size), n_batch)
            slices[-1] = np.arange(0, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        # item_set = set()
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            # item_set.update(set([t-1 for t in session]))
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
        # item_set = list(item_set)
        # index_list = [item_set.index(a) for a in self.targets[index]-1]

        return self.targets[index]-1, session_len,items, reversed_sess_item, mask
