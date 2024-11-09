import torch
from models import MNN_GNN, GNN, PathNN
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from surv_utils import cox_log_rank, CIndex_lifeline
import os
import scipy.sparse as sp
import math
from itertools import cycle
from datasets import Generic_MIL_Survival_Dataset, get_split_loader
from sksurv.metrics import concordance_index_censored
from tensorboardX import SummaryWriter
import time
import csv
import os



current_path = os.getcwd()

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None, device='cpu', source_dim=0, target_dim=0):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    # batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * source_dim + [[0]] * target_dim)).float().to(device)
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def distance_matrix(source, target, threshold=1000000):

    m, k = source.shape
    n, _ = target.shape

    if m*n*k < threshold:
        source = source.unsqueeze(1)
        target = target.unsqueeze(0)
        result = torch.sum((source - target) ** 2, dim=-1) ** (0.5)
    else:
        result = torch.empty((m, n))
        if m < n:
            for i in range(m):
                result[i, :] = torch.sum((source[i] - target)**2,dim=-1)**(0.5)
        else:
            for j in range(n):
                result[:, j] = torch.sum((source - target[j])**2,dim=-1)**(0.5)
    return result

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y
  


def mnn(source_feature, target_feature, topk=5):
    d_s_t = -distance_matrix(source_feature, target_feature)

    t_s_topk_index = d_s_t.topk(topk, dim=-1).indices
    s_t_topk_index = d_s_t.T.topk(topk, dim=-1).indices


    t_s_adjacency = torch.zeros((source_feature.shape[0], target_feature.shape[0]))
    s_t_adjacency = torch.zeros((target_feature.shape[0], source_feature.shape[0]))

    for i in range(source_feature.shape[0]):
        t_s_adjacency[i, t_s_topk_index[i]] = 1
    for j in range(target_feature.shape[0]):
        s_t_adjacency[j, s_t_topk_index[j]] = 1

    mnn_adjacency = t_s_adjacency * s_t_adjacency.T
    total_feature = torch.cat((source_feature, target_feature), dim=0)
    total_adj = torch.zeros((total_feature.shape[0], total_feature.shape[0]))
    total_adj[:source_feature.shape[0], source_feature.shape[0]:] = mnn_adjacency
    total_adj[source_feature.shape[0]:, :source_feature.shape[0]] = mnn_adjacency.T
    total_adj = total_adj + torch.eye(total_adj.shape[0])

    adj = sp.coo_matrix(total_adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))  
    adj = torch.LongTensor(indices)  
    return adj


@torch.no_grad()
def inference(args, model, loader):
    model.eval()
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_label = np.zeros((len(loader)))
    all_feature = None
    train_loss = 0
    for step, (img, _, _, _, _, _, _, label, surv, status) in enumerate(loader):
        if args.cuda:
            img, status = img.to(args.device), status.type(torch.FloatTensor).to(args.device)
            label = label.type(torch.LongTensor).to(args.device)
        feature, outputs, S = model(img)  # 前向传播
        if all_feature == None:
            all_feature = feature
        else:
            all_feature = torch.cat((all_feature, feature), dim=0)
        loss = nll_loss(outputs, S, label, status)
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[step] = risk
        all_censorships[step] = status.item()
        all_event_times[step] = surv
        all_label[step] = label.item()
        train_loss += loss.item()

    train_loss /= len(loader)

    return all_feature, all_risk_scores, all_censorships, all_event_times, all_label, train_loss

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024, device='cpu'):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim).to(device) for i in range(self.input_num)] #512,2

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

def train_EM(args, model, train_loader, optimizer, config, i):
    for _ in range(args.m - 1):
        model.train()
        all_risk_scores = np.zeros((len(train_loader)))
        all_censorships = np.zeros((len(train_loader)))
        all_event_times = np.zeros((len(train_loader)))

        dset_loaders = {}
        dset_loaders["source"] = DataLoader(config["source_dataset"], batch_size=args.batch_size, shuffle=False, drop_last=True)
        dset_loaders["target"] = DataLoader(config["target_dataset"], batch_size=args.batch_size, shuffle=False, drop_last=True)
        
        class_num = config['num_class']
        random_layer = RandomLayer([args.projection_size, class_num], config["loss"]["random_dim"], config['device'])
        ad_net = AdversarialNetwork(config["loss"]["random_dim"], 32).to(config['device'])
        total_correct = 0
        all_fc = None
        loss_params = config["loss"]
        step = 0
        for inputs_source, inputs_target, data in zip(cycle(dset_loaders["source"]), cycle(dset_loaders["target"]), train_loader):
            source_img, _, _, _, _, _, _, source_label, source_surv, source_status  = inputs_source
            target_img, _, _, _, _, _, _, target_label, target_surv, target_status= inputs_target
            data_img, _, _, _, _, _, _, data_label, data_surv, data_status  = data
            if args.cuda:
                source_img, source_status = source_img.to(args.device), source_status.to(args.device)
                target_img, target_status = target_img.to(args.device), target_status.to(args.device)
                data_img, data_status = data_img.to(args.device), data_status.to(args.device)
                source_label = source_label.type(torch.LongTensor).to(args.device)
                target_label = target_label.type(torch.LongTensor).to(args.device)
                data_label = data_label.type(torch.LongTensor).to(args.device)

            optimizer.zero_grad()
            source_img = source_img.view(-1, args.num_features)
            target_img = target_img.view(-1, args.num_features)
            source_feature, source_fc, source_S = model(source_img)
            target_feature, target_fc, target_S = model(target_img)
            feature_all, all_fc,  all_S = model(data_img)
            source_y = source_status
            target_y = target_status
            y_all = data_status
            surv_all = data_surv
            source_out = source_fc
            target_out = target_fc
            all_out = all_fc
            features = torch.cat((source_feature, target_feature), dim=0)
            outputs = torch.cat((source_out, target_out), dim=0)
            y =  torch.cat((source_y, target_y), dim=0)
            softmax_out = torch.nn.Softmax(dim=1)(outputs)
            entropy = Entropy(softmax_out)

            transfer_loss = CDAN([features, softmax_out], ad_net, entropy, calc_coeff(i),
                                random_layer,config['device'], source_feature.shape[0], target_feature.shape[0])

            transfer_loss.backward()

            source_feature, source_fc, source_S = model(source_img)
            target_feature, target_fc, target_S = model(target_img)
            feature_all, all_fc, all_S = model(data_img)
            source_y = source_status
            target_y = target_status
            y_all = data_status
            surv_all = data_surv
            source_out = source_fc
            target_out = target_fc
            all_out = all_fc
            features = torch.cat((source_feature, target_feature), dim=0)
            outputs = torch.cat((source_out, target_out), dim=0)
            y =  torch.cat((source_y, target_y), dim=0)
            softmax_out = torch.nn.Softmax(dim=1)(outputs)
            entropy = Entropy(softmax_out)

            transfer_loss = CDAN([features, softmax_out], ad_net, entropy, calc_coeff(i),
                                random_layer,config['device'], source_feature.shape[0], target_feature.shape[0])
            
            loss = nll_loss(all_fc, all_S, data_label, data_status)
            risk = -torch.sum(all_S, dim=1).detach().cpu().numpy()
            all_risk_scores[step] = risk
            all_censorships[step] = data_status.item()
            all_event_times[step] = data_surv
            step = step + 1
            classifier_loss = loss
            transfer_loss /= args.m
    
        two_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        two_loss.backward()
        optimizer.step()

        total_correct += int((outputs.argmax(dim=-1) == y).sum())

def first_brunch(args):
    print('Pre-training first brunch')

    # create results directory
    first_brunch_results_dir = "./first_brunch_results/{source_dataset}/[{target_dataset}]-[{time}]".format(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
    )
    if not os.path.exists(first_brunch_results_dir):
        os.makedirs(first_brunch_results_dir)

    # 5-fold cross validation
    header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
    best_epoch = ["best epoch"]
    best_score = ["best cindex"]

    for fold in range(5):

        writer_dir = os.path.join(first_brunch_results_dir, 'fold_' + str(fold))
        if not os.path.isdir(writer_dir):
            os.mkdir(writer_dir)
        writer = SummaryWriter(writer_dir, flush_secs=15)

        source_dataset = Generic_MIL_Survival_Dataset(
            csv_path="./csv/%s_all_clean.csv" % (args.source_dataset),
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=True,
            data_dir=args.source_dataset_dir,
            shuffle=False,
            seed=args.seed,
            patient_strat=False,
            n_bins=4,
            label_col="survival_months",
        )

        target_dataset = Generic_MIL_Survival_Dataset(
            csv_path="./csv/%s_all_clean.csv" % (args.target_dataset),
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=True,
            data_dir=args.target_dataset_dir,
            shuffle=False,
            seed=args.seed,
            patient_strat=False,
            n_bins=4,
            label_col="survival_months",
        )

        source_split_dir = os.path.join("./splits", args.k_folds, args.source_dataset)
        source_train_dataset, source_val_dataset = source_dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}.csv".format(source_split_dir, fold)
        )

        target_split_dir = os.path.join("./splits", args.k_folds, args.target_dataset)
        target_train_dataset, target_val_dataset = target_dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}.csv".format(target_split_dir, fold)
        )

        # source_train_dataset = Subset(source_train_dataset, list(range(20)))
        # target_val_dataset = Subset(target_val_dataset, list(range(20)))

        source_train_loader = get_split_loader(
            source_train_dataset,
            training=True,
            modal=args.modal,
            batch_size=args.batch_size,
        )


        target_val_loader = get_split_loader(
            target_val_dataset, modal=args.modal, batch_size=args.batch_size
        )

        print(
            "training: {}, validation: {}".format(len(source_train_dataset), len(target_val_dataset))
        )

        model = GNN(args, num_features=args.num_features, num_classes=args.prediction_size, conv_type=args.first_conv_type, 
                    pool_type=args.pool_type, emb=True, perturb_position=args.pp)
        
        if args.cuda:
            model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        val_best_score = 0
        val_best_epoch = 0

        for epoch in range(args.epochs):
            model.train()  # 设置为训练模式
            train_loss = 0.0
            source_all_risk_scores = np.zeros((len(source_train_loader)))
            source_all_censorships = np.zeros((len(source_train_loader)))
            source_all_event_times = np.zeros((len(source_train_loader)))
            source_train_loader = tqdm(source_train_loader, desc='Train Epoch: {}'.format(epoch))
            for step, (img, _, _, _, _, _, _, label, surv, status) in enumerate(source_train_loader):
                if args.cuda:
                    img, status = img.to(args.device), status.type(torch.FloatTensor).to(args.device)
                    label = label.type(torch.LongTensor).to(args.device)
                optimizer.zero_grad()  # 清空梯度
                _, outputs, S = model(img)  # 前向传播
                loss = nll_loss(outputs, S, label, status)
                risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                source_all_risk_scores[step] = risk
                source_all_censorships[step] = status.item()
                source_all_event_times[step] = surv
                train_loss += loss.item()
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重

            train_loss /= len(source_train_loader)
            c_index = concordance_index_censored((1-source_all_censorships).astype(bool),
                                                source_all_event_times, source_all_risk_scores, tied_tol=1e-08)[0]
                
            p_value = cox_log_rank(source_all_risk_scores, source_all_censorships, source_all_event_times)

            print('[train {}/{}] -'.format(epoch, args.epochs), 'Loss: {:.4f} -'.format(train_loss / (step + 1)),
                  'C-index:{:.4f}, p-value: {:.4f}'.format(c_index, p_value))
            
            if writer:
                writer.add_scalar('train/loss', train_loss, epoch)
                writer.add_scalar('train/c_index', c_index, epoch)

            # 验证模型
            model.eval()  # 设置为评估模式
            val_loss = 0.0
            target_all_risk_scores = np.zeros((len(target_val_loader)))
            target_all_censorships = np.zeros((len(target_val_loader)))
            target_all_event_times = np.zeros((len(target_val_loader)))
            target_val_loader = tqdm(target_val_loader, desc='Test Epoch: {}'.format(epoch))
            with torch.no_grad():
                for step, (img, _, _, _, _, _, _, label, surv, status) in enumerate(target_val_loader):
                    if args.cuda:
                        img, status = img.to(args.device), status.type(torch.FloatTensor).to(args.device)
                        label = label.type(torch.LongTensor).to(args.device)
                    _, outputs, S = model(img)  # 前向传播

                    loss = nll_loss(outputs, S, label, status)

                    risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                    target_all_risk_scores[step] = risk
                    target_all_censorships[step] = status.item()
                    target_all_event_times[step] = surv
                    val_loss += loss.item()

                # import pdb; pdb.set_trace()
                val_loss /= len(source_train_loader)
                c_index = concordance_index_censored((1-target_all_censorships).astype(bool),
                                                    target_all_event_times, target_all_risk_scores, tied_tol=1e-08)[0]
                p_value = cox_log_rank(target_all_risk_scores, target_all_censorships, target_all_event_times)
                if val_best_score < c_index:
                    val_best_score = c_index
                    val_best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(current_path, f'pretraining/first_{args.source_dataset}_{args.target_dataset}.pth'))
                    print(' *** best c-index={:.4f} at epoch {}'.format(val_best_score, val_best_epoch))

                print('[Test {}/{}] -'.format(epoch, args.epochs), 'Loss: {:.4f} -'.format(val_loss / (step + 1)),
                       'C-index:{:.4f}, p-value: {:.4f}'.format(c_index, p_value))
                
                if writer:
                    writer.add_scalar('val/loss', val_loss, epoch)
                    writer.add_scalar('val/c_index', c_index, epoch)

        best_epoch.append(val_best_epoch)
        best_score.append(val_best_score)

    best_epoch.append("~")
    best_epoch.append("~")
    best_score.append(np.mean(best_score[1:6]))
    best_score.append(np.std(best_score[1:6]))

    csv_path = os.path.join(first_brunch_results_dir, "results.csv")
    print("############", csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerow(best_epoch)
        writer.writerow(best_score)


def second_brunch(args):
    print('Pre-training second brunch')

    # create results directory
    second_brunch_results_dir = "./second_brunch_results/{source_dataset}/[{target_dataset}]-[{time}]".format(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
    )
    if not os.path.exists(second_brunch_results_dir):
        os.makedirs(second_brunch_results_dir)

    # 5-fold cross validation
    header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
    best_epoch = ["best epoch"]
    best_score = ["best cindex"]

    for fold in range(5):

        writer_dir = os.path.join(second_brunch_results_dir, 'fold_' + str(fold))
        if not os.path.isdir(writer_dir):
            os.mkdir(writer_dir)
        writer = SummaryWriter(writer_dir, flush_secs=15)

        source_dataset = Generic_MIL_Survival_Dataset(
            csv_path="./csv/%s_all_clean.csv" % (args.source_dataset),
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=True,
            data_dir=args.source_dataset_dir,
            shuffle=False,
            seed=args.seed,
            patient_strat=False,
            n_bins=4,
            label_col="survival_months",
        )

        target_dataset = Generic_MIL_Survival_Dataset(
            csv_path="./csv/%s_all_clean.csv" % (args.target_dataset),
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=True,
            data_dir=args.target_dataset_dir,
            shuffle=False,
            seed=args.seed,
            patient_strat=False,
            n_bins=4,
            label_col="survival_months",
        )

        source_split_dir = os.path.join("./splits", args.k_folds, args.source_dataset)
        source_train_dataset, source_val_dataset = source_dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}.csv".format(source_split_dir, fold)
        )

        target_split_dir = os.path.join("./splits", args.k_folds, args.target_dataset)
        target_train_dataset, target_val_dataset = target_dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}.csv".format(target_split_dir, fold)
        )

        # source_train_dataset = Subset(source_train_dataset, list(range(20)))
        # target_val_dataset = Subset(target_val_dataset, list(range(20)))

        source_train_loader = get_split_loader(
            source_train_dataset,
            training=True,
            modal=args.modal,
            batch_size=args.batch_size,
        )
        target_val_loader = get_split_loader(
            target_val_dataset, modal=args.modal, batch_size=args.batch_size
        )

        print(
            "training: {}, validation: {}".format(len(source_train_dataset), len(target_val_dataset))
        )


        model = PathNN(args, args.num_features, args.projection_size, args.cutoff, args.prediction_size, args.device, args.dropout, residuals=True, 
                       encode_distances=False, perturb_position=args.pp)
        
        if args.cuda:
            model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        val_best_score = 0
        val_best_epoch = 0

        for epoch in range(args.epochs):
            model.train()  # 设置为训练模式
            train_loss = 0.0
            source_all_risk_scores = np.zeros((len(source_train_loader)))
            source_all_censorships = np.zeros((len(source_train_loader)))
            source_all_event_times = np.zeros((len(source_train_loader)))
            source_train_loader = tqdm(source_train_loader, desc='Train Epoch: {}'.format(epoch))
            for step, (img, _, _, _, _, _, _, label, surv, status) in enumerate(source_train_loader):
                if args.cuda:
                    img, status = img.to(args.device), status.type(torch.FloatTensor).to(args.device)
                    label = label.type(torch.LongTensor).to(args.device)
                optimizer.zero_grad()  # 清空梯度
                _, outputs, S = model(img)  # 前向传播
                loss = nll_loss(outputs, S, label, status)

                risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                source_all_risk_scores[step] = risk
                source_all_censorships[step] = status.item()
                source_all_event_times[step] = surv
                train_loss += loss.item()
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重

            train_loss /= len(source_train_loader)
            c_index = concordance_index_censored((1-source_all_censorships).astype(bool),
                                                source_all_event_times, source_all_risk_scores, tied_tol=1e-08)[0]
                
            p_value = cox_log_rank(source_all_risk_scores, source_all_censorships, source_all_event_times)

            print('[train {}/{}] -'.format(epoch, args.epochs), 'Loss: {:.4f} -'.format(train_loss / (step + 1)),
                  'C-index:{:.4f}, p-value: {:.4f}'.format(c_index, p_value))
            
            if writer:
                writer.add_scalar('train/loss', train_loss, epoch)
                writer.add_scalar('train/c_index', c_index, epoch)

            # 验证模型
            model.eval()  # 设置为评估模式
            val_loss = 0.0
            target_all_risk_scores = np.zeros((len(target_val_loader)))
            target_all_censorships = np.zeros((len(target_val_loader)))
            target_all_event_times = np.zeros((len(target_val_loader)))
            target_val_loader = tqdm(target_val_loader, desc='Test Epoch: {}'.format(epoch))
            with torch.no_grad():
                for step, (img, _, _, _, _, _, _, label, surv, status) in enumerate(target_val_loader):
                    if args.cuda:
                        img, status = img.to(args.device), status.type(torch.FloatTensor).to(args.device)
                        label = label.type(torch.LongTensor).to(args.device)
                    _, outputs, S = model(img)  # 前向传播

                    loss = nll_loss(outputs, S, label, status)

                    risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                    target_all_risk_scores[step] = risk
                    target_all_censorships[step] = status.item()
                    target_all_event_times[step] = surv
                    val_loss += loss.item()

                # import pdb; pdb.set_trace()
                val_loss /= len(source_train_loader)
                c_index = concordance_index_censored((1-target_all_censorships).astype(bool),
                                                    target_all_event_times, target_all_risk_scores, tied_tol=1e-08)[0]
                p_value = cox_log_rank(target_all_risk_scores, target_all_censorships, target_all_event_times)
                if val_best_score < c_index:
                    val_best_score = c_index
                    val_best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(current_path, f'pretraining/second_{args.source_dataset}_{args.target_dataset}.pth'))
                    print(' *** best c-index={:.4f} at epoch {}'.format(val_best_score, val_best_epoch))

                print('[Test {}/{}] -'.format(epoch, args.epochs), 'Loss: {:.4f} -'.format(val_loss / (step + 1)),
                       'C-index:{:.4f}, p-value: {:.4f}'.format(c_index, p_value))
                
                if writer:
                    writer.add_scalar('val/loss', val_loss, epoch)
                    writer.add_scalar('val/c_index', c_index, epoch)

        best_epoch.append(val_best_epoch)
        best_score.append(val_best_score)

    best_epoch.append("~")
    best_epoch.append("~")
    best_score.append(np.mean(best_score[1:6]))
    best_score.append(np.std(best_score[1:6]))

    csv_path = os.path.join(second_brunch_results_dir, "results.csv")
    print("############", csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerow(best_epoch)
        writer.writerow(best_score)

def MNN_training(args, model, edge_index, feature, label, c):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(10):
        optimizer.zero_grad()
        pred, S = model(feature, edge_index)
        loss = nll_loss(pred, S, label, c)
        loss.backward()
        optimizer.step()

def EM_training(args):
    print("Doupling dual branches")
    # create results directory
    EM_first_results_dir = "./EM_first_results/{source_dataset}/[{target_dataset}]-[{time}]".format(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
    )
    if not os.path.exists(EM_first_results_dir):
        os.makedirs(EM_first_results_dir)

    EM_second_results_dir = "./EM_second_results/{source_dataset}/[{target_dataset}]-[{time}]".format(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
    )
    if not os.path.exists(EM_second_results_dir):
        os.makedirs(EM_second_results_dir)



    # 5-fold cross validation
    header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
    best_first_epoch = ["best epoch"]
    best_first_score = ["best cindex"]
    best_second_epoch = ["best epoch"]
    best_second_score = ["best cindex"]

    for fold in range(5):

        writer_dir1 = os.path.join(EM_first_results_dir, 'fold_' + str(fold))
        if not os.path.isdir(writer_dir1):
            os.mkdir(writer_dir1)
        writer1 = SummaryWriter(writer_dir1, flush_secs=15)

        writer_dir2 = os.path.join(EM_first_results_dir, 'fold_' + str(fold))
        if not os.path.isdir(writer_dir2):
            os.mkdir(writer_dir2)
        writer1 = SummaryWriter(writer_dir2, flush_secs=15)

        source_dataset = Generic_MIL_Survival_Dataset(
            csv_path="./csv/%s_all_clean.csv" % (args.source_dataset),
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=True,
            data_dir=args.source_dataset_dir,
            shuffle=False,
            seed=args.seed,
            patient_strat=False,
            n_bins=4,
            label_col="survival_months",
        )

        target_dataset = Generic_MIL_Survival_Dataset(
            csv_path="./csv/%s_all_clean.csv" % (args.target_dataset),
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=True,
            data_dir=args.target_dataset_dir,
            shuffle=False,
            seed=args.seed,
            patient_strat=False,
            n_bins=4,
            label_col="survival_months",
        )

        source_split_dir = os.path.join("./splits", args.k_folds, args.source_dataset)
        source_train_dataset, source_val_dataset = source_dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}.csv".format(source_split_dir, fold)
        )

        target_split_dir = os.path.join("./splits", args.k_folds, args.target_dataset)
        target_train_dataset, target_val_dataset = target_dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}.csv".format(target_split_dir, fold)
        )

        # source_train_dataset = Subset(source_train_dataset, list(range(10)))
        # target_val_dataset = Subset(target_val_dataset, list(range(10)))

        first_source_train_loader = get_split_loader(
            source_train_dataset,
            training=True,
            modal=args.modal,
            batch_size=args.batch_size,
        )
        first_target_val_loader = get_split_loader(
            target_val_dataset, modal=args.modal, batch_size=args.batch_size
        )

        e_step_cindex = []
        m_step_cindex = []
        args.method = 'first'

        model_first = GNN(args, num_features=args.num_features, num_classes=args.prediction_size, conv_type=args.first_conv_type, 
                    pool_type=args.pool_type, emb=True, perturb_position=args.pp)

        model_first.load_state_dict(torch.load(f'pretraining/first_{args.source_dataset}_{args.target_dataset}.pth', map_location=f'cuda:{args.device}'))

        if args.cuda:
            model_first = model_first.to(args.device)


        optimizer_first = torch.optim.Adam(model_first.parameters(), lr=args.lr, weight_decay=1e-4)

        args.method = 'second'
        second_source_train_loader = first_source_train_loader
        second_target_val_loader = first_target_val_loader

        model_second = PathNN(args, args.num_features, args.projection_size, args.cutoff, args.prediction_size, args.device, args.dropout, residuals=True, 
                       encode_distances=False, perturb_position=args.pp)
        model_second.load_state_dict(torch.load(f'pretraining/second_{args.source_dataset}_{args.target_dataset}.pth', map_location=f'cuda:{args.device}'))

        if args.cuda:
            model_second = model_second.to(args.device)
        optimizer_second = torch.optim.Adam(model_second.parameters(), lr=args.lr, weight_decay=1e-4)

        total_best_train_cindex = 0.

        config = {}
        config["loss"] = {"trade_off": 1.0}
        config['num_class'] = 4
        
        if torch.cuda.is_available():
            config['device'] = 'cuda:' + str(args.device)
        else:
            config['device'] = 'cpu'
        config["loss"]["random"] = args.random
        config["loss"]["random_dim"] = 1024

        ################## coupling 
        top_E, top_M = 0,0  
        for em_step in range(args.EM_epochs):
            if os.path.exists(f'./pretraining/M_second_{args.source_dataset}_{args.target_dataset}.pth'):
                model_second.load_state_dict(torch.load(f'pretraining/M_second_{args.dataset_name}_{args.target_dataset}.pth', map_location=f'cuda:{args.device}'))
            if args.cuda:
                model_second = model_second.to(args.device)
            source_second_feature, source_second_pred, source_second_label, source_second_surv, _, _ = inference(args, model_second, second_source_train_loader)
            target_second_feature, target_second_pred, target_second_label, target_second_surv, _, _ = inference(args, model_second, second_target_val_loader)

            '''
            统计label的占比
            '''
            args.e_threshold = np.mean(target_second_pred)
            args.time = np.mean(target_second_surv)
            condition1 = target_second_pred > args.e_threshold 
            condition2 = target_second_label == 1
            condition3 = target_second_surv < args.time

            condition4 = target_second_pred < args.e_threshold 
            condition5 = target_second_label == 0
            condition6 = target_second_surv > args.time

            condition7 = target_second_pred < args.e_threshold 
            condition8 = target_second_label == 1
            condition9 = target_second_surv > args.time

            idx1 = np.where(condition1 & condition2 & condition3)[0]
            idx2 = np.where(condition4 & condition5 & condition6)[0]
            idx3 = np.where(condition7 & condition8 & condition9)[0]
            pesudo_second_idx = np.concatenate((idx1, idx2, idx3))
            size_pesudo_second = pesudo_second_idx.shape[0]
            if size_pesudo_second <= 1:
                pesudo_second_idx = np.arange(2, dtype=np.int64)
                pesudo_second_label = target_second_pred[pesudo_second_idx]
                pesudo_second_surv = target_second_surv[pesudo_second_idx]
                ture_second_label = target_second_label[pesudo_second_idx]
                e_cindex = concordance_index_censored((1-ture_second_label).astype(bool),
                                    pesudo_second_surv, pesudo_second_label, tied_tol=1e-08)[0]
                print("e_cindex", e_cindex)
                e_step_cindex.append(e_cindex.item())
            else:
                pesudo_second_label = target_second_pred[pesudo_second_idx]
                pesudo_second_surv = target_second_surv[pesudo_second_idx]
                ture_second_label = target_second_label[pesudo_second_idx]
                e_cindex = concordance_index_censored((1-ture_second_label).astype(bool),
                                                    pesudo_second_surv, pesudo_second_label, tied_tol=1e-08)[0]
                print("e_cindex", e_cindex)
                e_step_cindex.append(e_cindex.item())

            # M step, generate mnn adj matrix for E step
            edge_index = mnn(source_second_feature, target_second_feature)

            # E step
            # source_dataset.status.values[pesudo_second_idx.cpu()] = pesudo_second_label.argmax(axis=-1)
            source_first = source_train_dataset
            target_first_copy = target_val_dataset
            E_training_data = source_first + Subset(target_first_copy, pesudo_second_idx)
            print('E step:',len(E_training_data), len(source_first), len(pesudo_second_idx))

            E_train_loader = get_split_loader(E_training_data, training=True, modal=args.modal, batch_size=args.batch_size)
            config["source_dataset"] = source_first
            config["target_dataset"] = Subset(target_first_copy, pesudo_second_idx)
            for i in range(1):
                train_EM(args, model_first, E_train_loader, optimizer_first, config, em_step)
                E_source_first_feature, E_source_pred, E_source_first_c, E_source_first_surv, E_source_first_label, E_source_loss = inference(args, model_first, first_source_train_loader)
                E_target_first_feature, E_target_pred, E_target_first_c, E_target_first_surv, E_target_first_label, _ = inference(args, model_first, first_target_val_loader)
                E_target_cindex = concordance_index_censored((1-E_target_first_label).astype(bool),
                                                    E_target_first_surv, E_target_pred, tied_tol=1e-08)[0]
                print("E_target_cindex", E_target_cindex)

                if total_best_train_cindex < E_target_cindex:
                    best_E_source_first_feature = E_source_first_feature
                    best_E_target_first_feature = E_target_first_feature
                    val_best_first_epoch = em_step
                    val_best_first_score = E_target_cindex
                    print(' *** best E_c-index={:.4f}'.format(E_target_cindex))
                    torch.save(model_first.state_dict(),
                            os.path.join(current_path, f'./pretraining/E_first_{args.source_dataset}_{args.target_dataset}_E.pth'))
                    
                if top_E < E_target_cindex:
                    top_E = E_target_cindex

            mnn_model = MNN_GNN(args, num_classes=4, conv_type=args.conv_type)

            if args.cuda:
                mnn_model = mnn_model.to(args.device)

            E_feature = torch.cat((best_E_source_first_feature, best_E_target_first_feature), dim=0)
            E_label = torch.cat((torch.from_numpy(E_source_first_label), torch.from_numpy(E_target_first_label)), dim=0).to(dtype=torch.long)
            E_surv = torch.cat((torch.from_numpy(E_source_first_surv), torch.from_numpy(E_target_first_surv)), dim=0)
            E_c = torch.cat((torch.from_numpy(E_source_first_c), torch.from_numpy(E_target_first_c)), dim=0)


            if args.cuda:
                E_label = E_label.to(args.device)
                E_c = E_c.to(args.device)
            MNN_training(args, mnn_model, edge_index, E_feature, E_label, E_c)

            E_pred, S = mnn_model(E_feature, edge_index)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            E_target_pred = risk[E_source_first_feature.shape[0]:]
            E_target_surv = E_surv[E_source_first_feature.shape[0]:].detach().cpu().numpy()
            E_label = E_label[E_source_first_feature.shape[0]:].detach().cpu().numpy()
            args.m_threshold = np.mean(E_target_pred)
            args.time = np.mean(E_target_surv)
            condition1 = E_target_pred > args.m_threshold 
            condition2 = E_label == 1
            condition3 = E_target_surv < args.time

            condition4 = E_target_pred < args.m_threshold 
            condition5 = E_label == 0
            condition6 = E_target_surv > args.time 

            condition7 = E_target_pred < args.m_threshold
            condition8 = E_label == 1
            condition9 = E_target_surv > args.time

            idx1 = np.where(condition1 & condition2 & condition3)[0]
            idx2 = np.where(condition4 & condition5 & condition6)[0]
            idx3 = np.where(condition7 & condition8 & condition9)[0]
            pesudo_first_idx = np.concatenate((idx1, idx2, idx3))
            size_pesudo_first = pesudo_first_idx.shape[0]       
            if size_pesudo_first <= 0:
                pesudo_first_idx = np.arange(2, dtype=np.int64)
                pesudo_first_label = E_target_pred[pesudo_first_idx]
                pesudo_first_surv = E_target_surv[pesudo_first_idx]
                ture_first_label = E_target_first_label[pesudo_first_idx]
                m_cindex = concordance_index_censored((1-ture_first_label).astype(bool),
                                    pesudo_first_surv, pesudo_first_label, tied_tol=1e-08)[0]
                print("m_cindex", m_cindex)
                m_step_cindex.append(m_cindex.item())
            else:
                pesudo_first_label = E_target_pred[pesudo_first_idx]
                pesudo_first_surv = E_target_surv[pesudo_first_idx]
                ture_first_label = E_target_first_label[pesudo_first_idx]
                m_cindex =  concordance_index_censored((1-ture_first_label).astype(bool),
                                                    pesudo_first_surv, pesudo_first_label, tied_tol=1e-08)[0]
                print("m_cindex", m_cindex)
                m_step_cindex.append(m_cindex.item())

            source_second = source_train_dataset
            target_second_copy = target_val_dataset
            
            M_training_data = source_second + Subset(target_first_copy, pesudo_first_idx)
            print('M step:',len(M_training_data),len(source_second),len(pesudo_first_idx))
            M_train_loader = get_split_loader(M_training_data, training=True, modal=args.modal, batch_size=args.batch_size)

            config["source_dataset"] = source_second
            config["target_dataset"] = Subset(target_first_copy, pesudo_first_idx)


            for i in range(1):
                train_EM(args, model_second, M_train_loader, optimizer_second, config, em_step)
                M_source_first_feature, M_source_pred, M_source_first_c, M_source_first_surv, M_source_first_label, M_source_loss = inference(args, model_second, first_source_train_loader)
                M_target_first_feature, M_target_pred, M_target_first_c, M_target_first_surv, M_target_first_label, _ = inference(args, model_second, first_target_val_loader)

                M_target_cindex = concordance_index_censored((1-M_target_first_label).astype(bool),M_target_first_surv, 
                                                             M_target_pred, tied_tol=1e-08)[0]
                print("M_target_cindex", M_target_cindex)

                if total_best_train_cindex < M_target_cindex:
                    print(' *** best M_c-index={:.4f}'.format(M_target_cindex))
                    val_best_second_epoch = em_step
                    val_best_second_score = M_target_cindex
                    torch.save(model_second.state_dict(),
                            os.path.join(current_path, f'./pretraining/M_second_{args.source_dataset}_{args.target_dataset}_M.pth'))
                    
                if top_M < M_target_cindex.item():
                    top_M = M_target_cindex.item()

        e_step_label_cindex = [round(num, 3) for num in e_step_cindex]
        m_step_label_cindex = [round(num, 3) for num in m_step_cindex]
        print(f'e_step_select_pseudo_cindex:{e_step_label_cindex}')
        print(f'm_step_select_pseudo_cindex:{m_step_label_cindex}')
        print(f"top_E, top_M: {top_E}, {top_M}")

        best_first_epoch.append(val_best_first_epoch)
        best_first_score.append(val_best_first_score)
        best_second_epoch.append(val_best_second_epoch)
        best_second_score.append(val_best_second_score)

    best_first_epoch.append("~")
    best_first_epoch.append("~")
    best_first_score.append(np.mean(val_best_first_score[1:6]))
    best_first_score.append(np.std(val_best_first_score[1:6]))

    best_second_epoch.append("~")
    best_second_epoch.append("~")
    best_second_score.append(np.mean(val_best_second_score[1:6]))
    best_second_score.append(np.std(val_best_second_score[1:6]))

    csv_path = os.path.join(EM_first_results_dir, "results.csv")
    print("############", csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerow(best_first_epoch)
        writer.writerow(best_first_score)

    csv_path = os.path.join(EM_second_results_dir, "results.csv")
    print("############", csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerow(best_second_epoch)
        writer.writerow(best_second_score)








