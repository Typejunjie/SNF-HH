import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from encoder import Autoencoder

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len


def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity

def pro_inputs(category,inputs):   #为种类设置特征变量，先读出每个序列对应的种类ID序列
    inputs_ID = []
    for item in inputs:
       if item == 0:
          inputs_ID += [0]
       else:
          inputs_ID += [category[item]]
    return inputs_ID 

class Data(Dataset):
    def __init__(self, data, category, avg_len, opt, train=True):

        inputs, mask, max_len = handle_data(data[0])
        self.category = category
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len
        self.avg_len = avg_len
        self.opt = opt
        self.train = train

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]

        if self.train == False:
            if sum(np.array(u_input) > 0) > self.opt.seq_len:
                # print(u_input)
                u_input, mask = filter(u_input, self.category, self.opt.ratio, self.opt.max_length, self.opt.var)
                # print('-----------------')
                # print(u_input)

        input_ID = pro_inputs(self.category, u_input)
        total = np.append(u_input, input_ID)
        total = total[total > 0]

        max_n_node = self.max_len
        node = np.unique(u_input)
        total_node = np.unique(total)
        if len(total_node)<max_n_node*2:
          total_node= np.append(total_node,0)
          
        items = node.tolist() + (max_n_node - len(node)) * [0]
        total_items = total_node.tolist() + (max_n_node * 2 - len(total_node)) * [0]
        total_adj = np.zeros((max_n_node*2, max_n_node * 2))

        for i in np.arange(len(u_input) - 1):
            u = np.where(total_node == u_input[i])[0][0]
            c = np.where(total_node == self.category[u_input[i]])[0][0]
            total_adj[u][u] = 1
            total_adj[c][c] = 4
            total_adj[u][c]= 2
            total_adj[c][u]= 3
            if u_input[i + 1] == 0:
                break          
            u2 = np.where(total_node == u_input[i + 1])[0][0]
            c2 = np.where(total_node == self.category[u_input[i + 1]])[0][0]
            total_adj[u][u2] = 1
            total_adj[u2][u] = 1
            
            total_adj[c][c2] = 4
            total_adj[c2][c] = 4

        alias_items = [np.where(total_node == i)[0][0] for i in u_input]
        alias_category = [np.where(total_node == i)[0][0] for i in input_ID]   #对应ID的相对位置
        
        return [torch.tensor(items),torch.tensor(mask), torch.tensor(target),
                torch.tensor(alias_items),torch.tensor(alias_category), 
                torch.tensor(total_adj),torch.tensor(total_items)]

    def __len__(self):
        return self.length


def get_N(items):

    nonzero_indices = torch.nonzero(items, as_tuple=False)
    last_nonzero_indices = torch.zeros(items.size(0), dtype=torch.long)
    for i in range(items.size(0)):
        row_indices = nonzero_indices[nonzero_indices[:, 0] == i, 1]
        if len(row_indices) > 0:
            last_nonzero_indices[i] = row_indices[-1]

    return max(last_nonzero_indices).item() + 1

def get_overlap(sessions):
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
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

def count_unique_paths(start, target, current_length, max_length, graph, memo):
    if current_length > max_length:
        return 0
    
    # 使用memo缓存已经计算过的路径
    if (start, target, current_length) in memo:
        return memo[(start, target, current_length)]
    
    path_count = 0
    if start == target and current_length > 0:
        path_count += 1
    
    for neighbor in graph[start]:
        path_count += count_unique_paths(neighbor, target, current_length + 1, max_length, graph, memo)
    
    # 缓存当前路径计算结果
    memo[(start, target, current_length)] = path_count
    return path_count

def path_count(nodes, max_length=5):
    unique_nodes, mapping_indices = np.unique(nodes, return_inverse=True)

    graph = defaultdict(list)
    for node in unique_nodes:
        graph[node] = []
    for i in range(len(nodes) - 1):
        graph[nodes[i]].append(nodes[i + 1])

    count = []
    total_unique_paths = 0
    # 初始化memo缓存字典
    memo = {}

    for i in unique_nodes:
        for start_node in unique_nodes:
            total_unique_paths += count_unique_paths(start_node, i, 0, max_length, graph, memo)
        count.append(total_unique_paths)
        total_unique_paths = 0

    return np.array(count)[mapping_indices]

def filter(input, category, ratio, max_length, var, flag='max', re_score=False):
    len_max = len(input)
    input = [i for i in input if i != 0]
    
    path_score = path_count(input, max_length)
    path_score = normalization(path_score, flag)

    cat_list = []
    for item in input:
        cat_list.append(category[item])
    unique_nodes, cat_indices = np.unique(cat_list, return_inverse=True)
    cat_score = normalization(unique_nodes, flag)[cat_indices]

    # 计算最终的score
    score = (path_score + cat_score) / 2
    # mean = np.mean(score)
    # std_dev = np.std(score)
    # # 计算变异系数
    # cv = (std_dev / mean) * 100

    # if np.var(score) * 100 > var:
    # # if cv > var:
    #     threshold = (max(score) - min(score)) * (ratio / 100) + min(score)
    #     # threshold = np.percentile(score, ratio)
    #     result = []
    #     for index, i in enumerate(score):
    #         if i > threshold:
    #             result.append(input[index])
    # else:
    #     result = input
    if re_score == True:
        return score

    N = 39 # 69 diginetica 39 Tmall 145 yoochoose1_64 29 Nowplaying
    mean = np.mean(score)
    data = np.concatenate([score, np.array([mean for i in range(N - len(score))])], axis=0).astype(float)
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    model = Autoencoder(N)
    model.load_state_dict(torch.load(f'./dict/autoencoder-Tmall.model'))
    model.eval()
    with torch.no_grad():
        reconstructed = model(data_tensor).numpy()

    reconstruction_error = np.abs(score - reconstructed[0][:len(score)]) * 100
    result = []
    for index, i in enumerate(reconstruction_error):
        if i > var and score[index] < mean:
            pass
        else:
            result.append(input[index])
    
    mask = np.concatenate([np.ones(len(result)), np.zeros(len_max - len(result))], axis=0)
    return np.concatenate([np.array(result), np.zeros(len_max - len(result))], axis=0).astype(int), mask

def get_N(items):
    N = 0
    items = items.numpy()
    for i in items:
        n = sum(i > 0)
        if n > N:
            N = n
    return N + 1

def normalization(array, flag='min-max'):
    if flag == 'min-max':
        min_s = min(array)
        max_s = max(array)
        if min_s == max_s:
            return np.ones_like(array)
        return (array - min_s) / (max_s - min_s)
    if flag == 'max':
        return array / max(array)
    
    if flag == 'softmax':
        exp_array = np.exp(array - np.max(array))  # 减去最大值是为了稳定计算，防止溢出
        return exp_array / np.sum(exp_array)

    raise ValueError("Unsupported normalization flag. Choose from ['min-max', 'max', 'softmax']")