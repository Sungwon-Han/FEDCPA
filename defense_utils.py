import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from geom_median.torch import compute_geometric_median
import torch.nn.utils.prune as prune
import scipy.stats

def get_krum(inputs, attacker_num=1):
    inputs = inputs.unsqueeze(0).permute(0, 2, 1)
    n = inputs.shape[-1]
    k = n - attacker_num - 2

    x = inputs.permute(0, 2, 1)

    cdist = torch.cdist(x, x, p=2)
    # find the k+1 nbh of each point
    nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
    # the point closest to its nbh
    i_star = torch.argmin(nbhDist.sum(2))
    mkrum = inputs[:, :, nbh[:, i_star, :].view(-1)].mean(2, keepdims=True)
    return mkrum, nbh[:, i_star, :].view(-1)

def get_foolsgold_score(total_score, grads, global_weight):
    n_clients = total_score.shape[0]
    norm_score = total_score
    
    wv = (norm_score - np.min(norm_score)) / (np.max(norm_score) - np.min(norm_score))
    wv[(wv == 1)] = .99
    wv[(wv == 0)] = .01
       
    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    model_weight_list = []
    for i in range(0, n_clients):
        if wv[i] != 0:
            current_weight = global_weight + wv[i] * grads[i] 
            model_weight_list.append(current_weight)
    fools_gold_weight = torch.cat(model_weight_list).mean(0, keepdims=True)
        
    return fools_gold_weight.view(-1), wv

def get_norm(inputs):
    number_to_consider = 8
    inputs = inputs.unsqueeze(0).permute(0, 2, 1)
    n = inputs.shape[-1]
    
    x = inputs.permute(0, 2, 1)
    norm = x.norm(2, dim=-1, keepdim=True)
    norm = norm.view(-1)
    sorted_norm, sorted_idx = torch.sort(norm)
    used_idx = sorted_idx[:number_to_consider]
    global_weight =  torch.mean(x[:, used_idx, :], dim=1).view(-1) 
    
    return global_weight, used_idx

def get_foolsgold(grads, global_weight):
    n_clients = grads.shape[0]
    grads_norm = F.normalize(grads, dim=1)
    cs = torch.mm(grads_norm, grads_norm.T)
    cs = cs - torch.eye(n_clients)
    maxcs, _ = torch.max(cs, axis=1)

    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    maxcs_2, _ = torch.max(cs, axis=1) 
    wv = 1 - maxcs_2
    

    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / torch.max(wv)
    wv[(wv == 1)] = .99
       
    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    
    model_weight_list = []
    for i in range(0, n_clients):
        if wv[i] != 0:
            current_weight = global_weight + wv[i]*grads[i] 
            model_weight_list.append(current_weight)
    fools_gold_weight = torch.cat(model_weight_list).mean(0, keepdims=True)
        
    return fools_gold_weight.view(-1), wv

def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output

def repeated_median(y):
    eps = np.finfo(float).eps
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


def reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh) + line_y * (reweight < thresh)
    return reweight_regulized, restricted_y

def weighted_average(w_list, weights):
    w_avg = copy.deepcopy(w_list[0])
    weights = weights / weights.sum()
    assert len(weights) == len(w_list)
    for k in w_avg.keys():
        w_avg[k] = 0
        for i in range(0, len(w_list)):
            w_avg[k] += w_list[i][k] * weights[i]
        # w_avg[k] = torch.div(w_avg[k], len(w_list))
    return w_avg, weights



def IRLS_aggregation_split_restricted(w_locals, LAMBDA=2, thresh=0.1):
    SHARD_SIZE = 2000
   
    w = []
    for net_id, net in enumerate(w_locals.values()):
        net_para = net.state_dict()
        w.append(net_para)
        
    w_med = copy.deepcopy(w[0])

    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)

        if total_num < SHARD_SIZE:
            reweight, restricted_y = reweight_algorithm_restricted(transposed_y_list, LAMBDA, thresh)
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y = reweight_algorithm_restricted(y, LAMBDA, thresh)
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...] = restricted_y

        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)
            
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)

    return w_med, reweight

def get_weight(model_weight):
    weight_tensor_result = []
    for k, v in model_weight.items():
        weight_tensor_result.append(v.reshape(-1).float())
    weights = torch.cat(weight_tensor_result)
    return weights

def get_weight_static(nets_this_round):
    model_weight_list = []
    net_id_list = []
    for net_id, net in enumerate(nets_this_round.values()):
        net_id_list.append(net_id)
        net_para = net.state_dict()
        model_weight = get_weight(net_para).unsqueeze(0)
        model_weight_list.append(model_weight)
    model_weight_cat = torch.cat(model_weight_list, dim=0)
    model_std, model_mean = torch.std_mean(model_weight_cat, unbiased=False, dim=0)
    
    return model_mean, model_std

def get_update_static(nets_this_round, global_net):
    model_weight_list = []
    net_id_list = []
    
    glboal_net_para = global_net.state_dict()
    global_weight = get_weight(glboal_net_para).unsqueeze(0)
    
    for net_id, net in enumerate(nets_this_round.values()):
        net_id_list.append(net_id)
        net_para = net.state_dict()
        model_weight = get_weight(net_para).unsqueeze(0)
        model_update = model_weight - global_weight
        model_weight_list.append(model_update)
    model_weight_cat = torch.cat(model_weight_list, dim=0)
    model_std, model_mean = torch.std_mean(model_weight_cat, unbiased=False, dim=0)
    
    return model_mean, model_std, model_weight_cat, global_weight

def global_aggregation(nets_this_round, args, fed_avg_freqs, global_w, party_list_this_round, prev_global_w, prev_prev_global_w, global_model, cur_round, logger):
    
    if args.global_defense == 'average':
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] / len(nets_this_round)
            else:
                for key in net_para:
                    global_w[key] += net_para[key] / len(nets_this_round)
                    
    elif args.global_defense == 'median':
        key_list = {}
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    key_list[key] = [net_para[key].unsqueeze(0)]
            else:
                for key in net_para:
                    key_list[key].append(net_para[key].unsqueeze(0))
        for key in net_para:
            key_value_cat = torch.cat(key_list[key])
            key_value_median, _ = torch.median(key_value_cat, dim=0)
            global_w[key] = key_value_median
            
    elif args.global_defense == 'krum':
        model_weight_list = []
        net_id_list = []
        for net_id, net in enumerate(nets_this_round.values()):
            net_id_list.append(net_id)
            net_para = net.state_dict()
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(model_weight)
        model_weight_cat = torch.cat(model_weight_list, dim=0)
        model_weight_krum, aggregate_idx = get_krum(model_weight_cat, args.expected_n_attacker)
        model_weight_krum = model_weight_krum.reshape(-1)
        
        aggregate_idx_list = torch.tensor(party_list_this_round)[aggregate_idx].tolist()
        aggregate_idx_list.sort()
        removed_idx = list(set(party_list_this_round) - set(aggregate_idx_list))
        logger.info(">> Removed Network IDX: {}".format(' '.join(map(str, removed_idx))))

        current_idx = 0
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_w[key] = model_weight_krum[current_idx:current_idx+length].reshape(net_para[key].shape)
            current_idx +=length 
    
    elif args.global_defense == 'foolsgold':
        model_weight_list = []
        net_id_list = []
        for net_id, net in enumerate(nets_this_round.values()):
            net_id_list.append(net_id)
            net_para = net.state_dict()
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(model_weight)
        model_weight_cat= torch.cat(model_weight_list, dim=0)
        
        update_mean, update_std, update_cat, global_weight = get_update_static(nets_this_round, global_model)
        model_weight_foolsgold, wv  = get_foolsgold(update_cat, global_weight)
        
        logger.info(">> Network Weight: {}".format(' '.join(map(str, wv.tolist()))))
        
        current_idx = 0
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_w[key] = model_weight_foolsgold[current_idx:current_idx+length].reshape(net_para[key].shape)
            current_idx +=length 
    
    elif args.global_defense == 'residual':
        model_weight_list = []
        net_id_list = []
        global_w, reweight = IRLS_aggregation_split_restricted(nets_this_round, 2.0, 0.05)
        logger.info(">> Network Weight: {}".format(' '.join(map(str, reweight.tolist()))))
    
    elif args.global_defense == 'trimmed_mean':
        net_para_list = []
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            net_para_list.append(net_para)
            
        trimmed_num = 1
        
        # Trimmed mean
        for key in global_w:
            net_para_stack = torch.stack([net_row[key] for net_row in net_para_list])
            net_shape = net_para_stack.shape[1:]
            net_para_stack = net_para_stack.reshape(len(net_para_list), -1)
            net_para_sorted = net_para_stack.sort(dim=0).values
            result = net_para_sorted[trimmed_num:-trimmed_num, :]
            result_type = result.dtype
            result = result.float().mean(dim=0).type(result_type)
            result = result.reshape(net_shape)
            global_w[key] = result
            
    elif args.global_defense == 'norm':
        model_weight_list = []
        net_id_list = []
        for net_id, net in enumerate(nets_this_round.values()):
            net_id_list.append(net_id)
            net_para = net.state_dict()
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(model_weight)
        model_weight_cat = torch.cat(model_weight_list, dim=0)
        model_weight_norm, aggregate_idx = get_norm(model_weight_cat)
        
        aggregate_idx_list = torch.tensor(party_list_this_round)[aggregate_idx].tolist()
        aggregate_idx_list.sort()
        removed_idx = list(set(party_list_this_round) - set(aggregate_idx_list))
        logger.info(">> Removed Network IDX: {}".format(' '.join(map(str, removed_idx))))

        current_idx = 0
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_w[key] = model_weight_norm[current_idx:current_idx+length].reshape(net_para[key].shape)
            current_idx +=length
    
    elif args.global_defense == 'rfa':
        model_weight_list = []
        net_id_list = []
        for net_id, net in enumerate(nets_this_round.values()):
            net_id_list.append(net_id)
            net_para = net.state_dict()
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(list(model_weight))
        
        model_weight_rfa = compute_geometric_median(model_weight_list, weights=None).median[0]

        current_idx = 0
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_w[key] = model_weight_rfa[current_idx:current_idx+length].reshape(net_para[key].shape)
            current_idx +=length 
        
    elif args.global_defense == 'cpa':  
        local_global_w_list = []
        
        global_para = global_model.state_dict()
        global_critical_dict = {}
        for name, val in global_para.items():
            if val.dim() in [2, 4]:
                critical_weight = torch.abs((prev_global_w[name] - prev_prev_global_w[name]) * prev_global_w[name])
                global_critical_dict[name] = critical_weight
        
        global_w_stacked = get_weight(global_critical_dict).view(1, -1)      
        global_topk_indices = torch.abs(global_w_stacked).topk(int(global_w_stacked.shape[1] * 0.01)).indices
        global_bottomk_indices = torch.abs(global_w_stacked).topk(int(global_w_stacked.shape[1] * 0.01), largest=False).indices
        
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            critical_dict = {}
            for name, val in net_para.items():
                if val.dim() in [2, 4]:
                    critical_weight = torch.abs((val - prev_global_w[name]) * val)
                    critical_dict[name] = critical_weight

            local_global_w_list.append(get_weight(critical_dict))
        w_stacked = torch.stack(local_global_w_list, dim=0)
        local_topk_indices = torch.abs(w_stacked).topk(int(w_stacked.shape[1] * 0.01)).indices
        local_bottomk_indices = torch.abs(w_stacked).topk(int(w_stacked.shape[1] * 0.01), largest=False).indices

        pairwise_score = np.zeros((len(nets_this_round), len(nets_this_round)))
        for i in range(len(nets_this_round)):
            for j in range(len(nets_this_round)):
                if i == j:
                    pairwise_score[i][j] = 1
                elif i < j:
                    continue       

                topk_intersection = list(set(local_topk_indices[i].tolist()) & set(local_topk_indices[j].tolist()))
                topk_corr_dist = ((scipy.stats.pearsonr(w_stacked[i, topk_intersection].numpy(), w_stacked[j, topk_intersection].numpy())[0]) + 1) / 2
                topk_jaccard_dist = len(topk_intersection) / (len(local_topk_indices[i]) + len(local_topk_indices[j]) - len(topk_intersection))

                bottomk_intersection = list(set(local_bottomk_indices[i].tolist()) & set(local_bottomk_indices[j].tolist()))
                bottomk_corr_dist = ((scipy.stats.pearsonr(w_stacked[i, bottomk_intersection].numpy(), w_stacked[j, bottomk_intersection].numpy())[0]) + 1) / 2
                bottomk_jaccard_dist = len(bottomk_intersection) / (len(local_bottomk_indices[i]) + len(local_bottomk_indices[j]) - len(bottomk_intersection))        

                pairwise_score[i][j] = (topk_corr_dist + bottomk_corr_dist) / 2 + (topk_jaccard_dist + bottomk_jaccard_dist) / 2
                pairwise_score[j][i] = (topk_corr_dist + bottomk_corr_dist) / 2 + (topk_jaccard_dist + bottomk_jaccard_dist) / 2
                
        global_score = np.zeros(len(nets_this_round))
        for i in range(len(nets_this_round)):

                topk_intersection = list(set(local_topk_indices[i].tolist()) & set(global_topk_indices[0].tolist()))
                topk_corr_dist = ((scipy.stats.pearsonr(w_stacked[i, topk_intersection].numpy(), global_w_stacked[0, topk_intersection].numpy())[0]) + 1) / 2
                topk_jaccard_dist = len(topk_intersection) / (len(local_topk_indices[i]) + len(global_topk_indices[0]) - len(topk_intersection))

                bottomk_intersection = list(set(local_bottomk_indices[i].tolist()) & set(global_bottomk_indices[0].tolist()))
                bottomk_corr_dist = ((scipy.stats.pearsonr(w_stacked[i, bottomk_intersection].numpy(), global_w_stacked[0, bottomk_intersection].numpy())[0]) + 1) / 2
                bottomk_jaccard_dist = len(bottomk_intersection) / (len(local_bottomk_indices[i]) + len(global_bottomk_indices[0]) - len(bottomk_intersection))        

                global_score[i]= (topk_corr_dist + bottomk_corr_dist) / 2 + (topk_jaccard_dist + bottomk_jaccard_dist) / 2
                
        remove_num = int(len(nets_this_round) * 0.8)
        total_score = np.mean(pairwise_score, axis=1) + global_score
        
        update_mean, update_std, update_cat, global_weight = get_update_static(nets_this_round, global_model)
        model_weight_foolsgold, wv  = get_foolsgold_score(total_score, update_cat, global_weight)
        
        logger.info(">> Network Weight: {}".format(' '.join(map(str, wv.tolist()))))
        
        current_idx = 0
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_w[key] = model_weight_foolsgold[current_idx:current_idx+length].reshape(net_para[key].shape)
            current_idx += length
            
    return global_w