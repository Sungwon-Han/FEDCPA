import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import torch.nn.utils.prune as prune
import scipy.stats
from sklearn.mixture import GaussianMixture
import datetime
from functools import reduce
from model import *
from utils import *
from defense_utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=20, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')    
    parser.add_argument('--global_defense', type=str, default='cpa',help='communication strategy')
    
    # poison settings
    parser.add_argument('--attacker_type', type=str, default='untargeted_flip', help='attacker type (either untargeted_gaussian untargeted_flip)')
    parser.add_argument('--attacker_ratio', type=float, default=0.2, help='ratio for number of attackers')
    parser.add_argument('--noise_ratio', type=float, default=0.8, help='noise ratio for label flipping (0 to 1)')
      
    return parser.parse_args()


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'cifar10', 'svhn'}:
        n_classes = 10
    elif args.dataset == 'tinyimagenet':
        n_classes = 200

    for net_i in range(n_parties):
        if args.dataset in {'cifar10', 'svhn'}:
            net = ResNet18_cifar10()
        elif args.dataset == 'tinyimagenet':
            net = ResNet18_tinyimage()

        if device == 'cpu':
            net.to(device)
        else:
            net = net.cuda()
        nets[net_i] = net


    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    net.cuda()
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0
    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    net.to('cpu')

    return train_acc, test_acc


def local_train_net(nets, args, net_dataidx_map, attacker_id_list=[], train_dl=None, 
                    test_dl=None, global_model = None, round=None, device="cpu"):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_w = global_model.state_dict()
        global_weight = get_weight(global_w).unsqueeze(0)

    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]

        if net_id in attacker_id_list:
            prefix = 'attacker'
            if args.attacker_type == 'untargeted_flip':
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, 
                                                                     attacker_type=args.attacker_type, noise_ratio=args.noise_ratio,
                                                                     perturb_probs=perturb_prob_dict[net_id]) 
            elif args.attacker_type == 'untargeted_gaussian':
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)    
            
        else:
            prefix = 'normal'
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs
        
        if args.attacker_type == "untargeted_gaussian":
            if net_id in attacker_id_list:
                weight_std = torch.std(global_weight)
                model_attack = global_weight + 0.05 * weight_std * torch.randn(global_weight.size())
                model_attack = model_attack.squeeze(0)
                attacker_w = net.state_dict()

                current_idx = 0
                for key in attacker_w:
                    length = len(attacker_w[key].reshape(-1))
                    attacker_w[key] = model_attack[current_idx : current_idx + length].reshape(attacker_w[key].shape)
                    current_idx += length
                net.load_state_dict(attacker_w)
                nets_this_round[net_id] = net
            else:
                trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, device=device)
                logger.info(prefix + " net %d final test acc %f" % (net_id, testacc))
        else:
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, device=device)
            logger.info(prefix + " net %d final test acc %f" % (net_id, testacc))
    
    if global_model:
        global_model.to('cpu')
    return nets


args = get_args()
mkdirs(args.logdir)
mkdirs(args.modeldir)

device = torch.device(args.device)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

if args.log_file_name is None:
    args.log_file_name = f'{args.dataset}-beta_{args.beta}-num-party_{args.n_parties}-local-epoch_{args.epochs}-type_{args.attacker_type}-attacker-ratio_{args.attacker_ratio}-noise_{args.noise_ratio}_{args.global_defense}'
log_path = args.log_file_name + '.log'
logging.basicConfig(
    filename=os.path.join(args.logdir, log_path),
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger.info(device)

seed = args.init_seed
logger.info("#" * 100)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

logger.info("Partitioning data")
X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
    args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

n_party_per_round = min(args.n_parties, 10)
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []
if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        this_round = random.sample(party_list, n_party_per_round)
        this_round.sort()
        party_list_rounds.append(this_round)
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)


n_attacker = int(args.attacker_ratio * args.n_parties)
expected_n_attacker = int(args.attacker_ratio * n_party_per_round)
args.expected_n_attacker = expected_n_attacker - 1

attacker_id_list = random.sample(party_list, n_attacker)   

    
logger.info(">> Attacker Network IDX: {}".format(' '.join(map(str, attacker_id_list))))

    
n_classes = len(np.unique(y_train))

train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                           args.datadir,
                                                                           args.batch_size,
                                                                           32)
print("len train_dl_global:", len(train_ds_global))
train_dl=None
data_size = len(test_ds_global)

logger.info("Initializing nets")
nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
global_model = global_models[0]
n_comm_rounds = args.comm_round
    
perturb_prob_dict = {}
if args.attacker_type == 'untargeted_flip':
    for attacker_id in attacker_id_list:
        perturb_prob_dict[attacker_id] = np.random.dirichlet(np.repeat(0.25, n_classes))
else:
    perturb_prob_dict = None
    
    
for round in range(n_comm_rounds):
    logger.info("in comm round:" + str(round))
    party_list_this_round = party_list_rounds[round]

    global_w = global_model.state_dict()
    
    
    if round != 0:
        prev_prev_global_w = copy.deepcopy(prev_global_w)
        prev_global_w = copy.deepcopy(global_model.state_dict())
    
        
    nets_this_round = {k: nets[k] for k in party_list_this_round}
    for net in nets_this_round.values():
        net.load_state_dict(global_w)
        
    local_train_net(nets_this_round, args, net_dataidx_map, attacker_id_list=attacker_id_list, 
                    train_dl=train_dl, test_dl=test_dl, global_model= global_model, device=device)

    total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
    
    if round == 0:
        prev_prev_global_w = copy.deepcopy(global_model.state_dict())
        prev_global_w = copy.deepcopy(global_model.state_dict())
        
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    prev_global_w[key] = net_para[key] / len(nets_this_round)
            else:
                for key in net_para:
                    prev_global_w[key] += net_para[key] / len(nets_this_round)
    
    
    
    global_w = global_aggregation(nets_this_round, args, fed_avg_freqs, global_w, party_list_this_round, prev_global_w, prev_prev_global_w, global_model, round, logger)
    global_model.load_state_dict(global_w)
    global_model.cuda()
    
    test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

    logger.info('>> Global Model Test accuracy: %f' % test_acc)
    global_model.to('cpu')
    
mkdirs(args.modeldir+'fedavg/')
torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')