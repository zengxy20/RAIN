import time
from util import Data
from model import *
import torch
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,default='Tmall',help='dataset name: Retailrocket/diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=1000, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=10, help='the number of layer used')
parser.add_argument('--lambda1', type=float, default=0.0001, help='diff task maginitude')
parser.add_argument('--lambda2', type=float, default=1, help='ssl task maginitude')#1000000
parser.add_argument('--gam', type=float, default=0.1, help='cofident ratio')
parser.add_argument('--numcuda', type=int, default=2,help='which GPU train')

opt = parser.parse_args()
print(opt)


def main():
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    torch.cuda.manual_seed_all(1000)
    np.random.seed(2023)
    random.seed(2023)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.numcuda)
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    all_train = pickle.load(open('datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
        opt.beta = 1
    elif opt.dataset == 'Nowplaying':
        n_node = 60416
    elif opt.dataset == 'Retailrocket':
        n_node = 60416
        opt.beta = 10
    else:
        n_node = max_item+1

    patience = 0
    train_d = Data(train_data, all_train, opt.gam, shuffle=False, n_node=n_node)
    test_d = Data(test_data, all_train, opt.gam, shuffle=False, n_node=n_node)
    model = trans_to_cuda(
        RAIN(adjacency=train_d.adjacency, adjacency_a=train_d.adjacency_a, adjacency_1=train_d.adjacency_1,ad1r=train_d.ad1r, ad1c=train_d.ad1c, n_node=n_node, lr=opt.lr, lambda1=opt.lambda1, lambda2=opt.lambda2, 
             layers=opt.layer, emb_size=opt.embSize, batch_size=opt.batchSize, dataset=opt.dataset))
    top_K = [1, 5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        if patience>=10:
            break
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_d, test_d, epoch)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                if K == 20:
                    torch.save(model.state_dict(), 'datasets/' + opt.dataset +  '/'+str(opt.gam) + '_besthit_model_parameter.tar')
                    patience = 0
            else:
                if K == 20:
                    patience+=1
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        metrics['AUC'] = np.mean(metrics['AUC']) * 100

        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                    (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                    best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))

if __name__ == '__main__':
    main()
