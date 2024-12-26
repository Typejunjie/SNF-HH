import time
import argparse
import pickle
from model import *
from utils import *
import os

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='diginetica/Nowplaying/Tmall/yoochoose1_64')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=12)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--layers', type=int, default=5)

# filter parameters
parser.add_argument('--ratio', type=int, default=1)
parser.add_argument('--max_length', type=int, default=3)
parser.add_argument('--seq_len', type=int, default=4)
parser.add_argument('--var', type=float, default=27)
# dig 1, 3, 10, 23
# Tmall 10, 3, 4, 27
opt = parser.parse_args()

def device():
    if torch.cuda.is_available():
        # return variable
        return 'cuda'
    else:
        return 'cpu'

def main():
    if opt.dataset == 'diginetica':
        init_seed(2020)
        num_node = 43098
        opt.n_iter = 2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.0
        n_category = 996    #(995+1)
        avg_len = 5.12
    elif opt.dataset == 'Nowplaying':
        init_seed(2020)
        num_node = 60417
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
        n_category = 11462 #(11461 + 1)
        avg_len = 7.42
    elif opt.dataset == 'Tmall':
        init_seed(2024)
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
        n_category = 712 #711 + 1
        avg_len = 6.69
    elif opt.dataset == 'yoochoose1_64':
        init_seed(2024)
        num_node = 37484
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
        n_category = 61 # 60 + 1
        avg_len = 6.16
    else:
        raise KeyError

    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
        
    category = pickle.load(open('datasets/' + opt.dataset + '/category.txt', 'rb'))   #读出商品的类别信息

    test_data = Data(test_data, category, avg_len, opt, False)

    model = trans_to_cuda(CombineGraph(opt, num_node+n_category-1, n_category, category))
    if device() == 'cuda':
        model.load_state_dict(torch.load(f'./dict/{opt.dataset}-parameters.model'))
    else:
        model.load_state_dict(torch.load(f'./dict/{opt.dataset}-parameters.model', map_location=torch.device('cpu')))
    print(opt)

    def test(model, test_data, topk):
        model.eval()
        test_loader = torch.utils.data.DataLoader(test_data, num_workers=0, batch_size=model.batch_size,
                                                shuffle=False, pin_memory=True)
        hit, mrr = [[] for i in topk], [[] for i in topk]
        total_loss = 0.0

        for data in tqdm(test_loader, colour='green', desc='Estimating', leave=False):
            # targets, scores, con_loss = forward(model, data)
            targets, scores = forward(model, data)
            loss = model.loss_function(scores, trans_to_cuda(targets).long() - 1)
            # loss = loss + con_loss
            total_loss += loss.item()
            targets = targets.numpy()

            for index, i in enumerate(topk):
                sub_scores = scores.topk(i)[1]
                sub_scores = trans_to_cpu(sub_scores).detach().numpy()
                for score, target in zip(sub_scores, targets):
                    hit[index].append(np.isin(target - 1, score))
                    if len(np.where(score == target - 1)[0]) == 0:
                        mrr[index].append(0)
                    else:
                        mrr[index].append(1 / (np.where(score == target - 1)[0][0] + 1))
        for index, j in enumerate(topk):
            hit[index] = np.mean(hit[index]) * 100
            mrr[index] = np.mean(mrr[index]) * 100
        return hit, mrr

    topk = [5, 10, 20]

    hit, mrr = test(model, test_data, topk)
    # for index, i in enumerate(topk):
    #     if hit[index] >= best_result[index][0]:
    #         best_result[index][0] = hit[index]
    #     if mrr[index] >= best_result[index][1]:
    #         best_result[index][1] = mrr[index]

    print('Result:')
    for index, i in enumerate(topk):
        print(f'\tRecall@{i}:\t%.4f\tMMR@{i}:\t%.4f\t'% (hit[index], mrr[index]))

    return [hit, mrr]


if __name__ == '__main__':

    # p = [i for i in range(4, 21)]
    # all_results = []
    # filename = f'./data/{opt.dataset}seq_len_4-20.pkl'
    # # filename = f'./data/{opt.dataset}max_len_3-20.pkl'
    # for i in p:
    #     opt.seq_len = i
    #     result = main() 
    #     all_results.append(result)
    #     with open(filename, 'wb') as f:
    #         pickle.dump(all_results, f)

    # p = [i for i in range(10, 31)]
    # for i in p:
    #     opt.var = i
    #     main()

    main()
