import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from ae import autoencoder
from torchvision.utils import save_image
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.metrics import accuracy_score
import faiss
import matplotlib.pyplot as plt 

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if args.da == 'oda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()
    elif args.da == 'pda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes or int(reci[1]) == len(args.src_classes):
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()
    args.out_file.write(' '.join(txt_tar) + '\n')
    args.out_file.flush()
    args.out_file.write('text tar length: ' + str(len(txt_tar)) + '\n')
    args.out_file.flush()
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def saveConfMat(all_label, predict):
    pos_labels = np.arange(0,args.class_num+1)
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float(), labels = pos_labels)
    print(matrix.shape)
    font = {'family' : 'normal',
            'size'   : 2}
    plt.rc('font', **font)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.savefig(args.output_dir + "/"+ str(args.data_mode) + "_pda_confusion_matrix.png", dpi=400)
    plt.close()

def cal_acc(loader, faiss_index, netF, netB, netC, pt_fe, optimal_threshold, flag=False, threshold=0.1):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            pt_feas = pt_fe(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_feas = feas.cpu()
                all_label = labels.float()
                all_pt_feas = pt_feas.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_feas = torch.cat((all_feas, feas.cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_pt_feas = torch.cat((all_pt_feas, pt_feas.float().cpu()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    # anom det accuracy
    if flag:
        all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)

        # perform feature space anomaly detection
        D, _ = faiss_index.search(all_pt_feas.cpu().detach().numpy(), 1)
        ano_score = np.sum(D, axis=1)
        #norm_ano_score = (ano_score - np.amin(ano_score)) / (np.amax(ano_score)-np.amin(ano_score))
        predict[ano_score > optimal_threshold] = args.class_num

        # AD labels
        anom_gt = np.ones_like(all_label)
        anom_gt[all_label<args.class_num] = 0
        auc = roc_auc_score(anom_gt, ano_score)
        print("Current Iter AUC: ", auc)

        target_classes = args.tar_range_end - args.tar_range_start

        #matrix = confusion_matrix(all_label, torch.squeeze(predict).float())[args.tar_range_start:,-target_classes:]
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())

        saveConfMat(all_label, predict)
        

        diag = matrix.diagonal()[args.tar_range_start:args.tar_range_end]
        sums = matrix.sum(axis=1)[args.tar_range_start:args.tar_range_end]
        # first_dim_sums = matrix.sum(axis=1)
        # cur_diag = np.concatenate((diag[args.tar_range_start,args.tar_range_end]))
        # curr_sum = np.concatenate((first_dim_sums[args.tar_range_start,first_dim_sums.tar_range_end]))
        
        acc = np.zeros_like(diag)
        print(diag)
        print(sums)
        for i in range(target_classes):
            if sums[i] == 0:
                continue
            else:
                acc[i] = diag[i]/sums[i]*100
        print(acc)
        aacc = np.mean(acc)
        print(aacc)

        print("mat shape", matrix.shape)
        #acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        print("mat diag", diag)
        print("mat curr_sum", sums)
        print("min pred", torch.min(predict))
        print("max pred", torch.max(predict))
        print("min label", torch.min(all_label))
        print("max label", torch.max(all_label))
        #aacc = np.mean(acc[:-1])
        #aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def train_target(args):
    dset_loaders = data_load(args)

    # get pre-trained VIT
    vit = torchvision.models.vit_l_16(pretrained=False)
    state_dict = torch.load('./vit_l_16-852ce7e3.pth')
    status = vit.load_state_dict(state_dict)
    print("loaded vit")
    vit.eval()
    vit.cuda()

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    vit.encoder.ln.register_forward_hook(get_activation('ln'))


    ## set base network
    if args.net[0:3] == 'res':
        resnet50 = torchvision.models.resnet50(pretrained=False)
        state_dict = torch.load('./resnet50-0676ba61.pth')
        status = resnet50.load_state_dict(state_dict)
        print("Loaded Resnet50")
        netF = network.ResBase(res_name=args.net, existingModel = resnet50).cuda()
        print("Finished Loading Resnet50")
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    tt = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label, faiss_index, optimal_threshold = obtain_label_L2_confident(dset_loaders['test'], netF, netB, netC, vit, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        pred = mem_label[tar_idx]
        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))

        with torch.no_grad():
            ad_features = vit(inputs_test)
            # perform feature space anomaly detection
            D, _ = faiss_index.search(ad_features.cpu().detach().numpy(), 1)
            ano_score = np.sum(D, axis=1)
            #norm_ano_score = (ano_score - np.amin(ano_score)) / (np.amax(ano_score)-np.amin(ano_score))
            pred[ano_score > optimal_threshold] = args.class_num # set detected anomalies to outlier class in pseudo label GT
        outputs_test = netC(features_test)

        softmax_out = nn.Softmax(dim=1)(outputs_test)
        # filtering out detected anomaly indices from prediction and pseudo labels GT
        outputs_test_known = outputs_test[pred < args.class_num, :]
        pred = pred[pred < args.class_num]

        if len(pred) == 0:
            print("all anomalies", tt)
            del features_test
            del outputs_test
            tt += 1
            continue

        if args.cls_par > 0:
            classifier_loss = nn.CrossEntropyLoss()(outputs_test_known, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out_known = nn.Softmax(dim=1)(outputs_test_known)
            entropy_loss = torch.mean(loss.Entropy(softmax_out_known))
            if args.gent:
                msoftmax = softmax_out_known.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            acc_s_te, _ = cal_acc(dset_loaders['test'], faiss_index, netF, netB, netC, vit, optimal_threshold, True)            
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()

    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        
    return netF, netB, netC

def obtain_label_L2_confident(loader, netF, netB, netC, pt_fe, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            pt_feas = pt_fe(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_pt_feas = pt_feas.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_pt_feas = torch.cat((all_pt_feas, pt_feas.float().cpu()), 0)
    orig_all_label = all_label.clone()
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()

    print("Entropy Shape", ent.shape)

    # use UDA pseudo labeling past this point

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    # build pre-trained feature set for each shared class using source classifier entropy
    # For each class, find top 5 highest entropy as outlier set 
    # for each class, find top 5 lowest entropy as inlier set
    start = True
    for cls in range(args.tar_range_start, args.tar_range_end):
        all_idx = np.arange(0, ent.shape[0])
        cls_idx = np.where(predict == cls)[0]
        if cls_idx.shape[0] == 0:
            print("No confident predictions for class", cls, "with cls idx shape: ", cls_idx.shape)
            continue
        cls_ent = ent[cls_idx]
        cls_idx_ent = sorted(list(zip(cls_idx, cls_ent)), key = lambda x: x[1]) # sort by entropy within class (low to high)
        #print(len(list(zip(*cls_idx_ent))))
        cls_idx_ent_sorted = list(zip(*cls_idx_ent))[0] # unzip and get indices of each class with lowest entropy (up to 5)
        
        all_idx_ent = sorted(list(zip(all_idx, ent)), key = lambda x: x[1])
        all_idx_ent_sorted = list(zip(*all_idx_ent))[0] # unzip and get indices of each class with lowest entropy (up to 5)

        high_ent_idx = list(all_idx_ent_sorted[-args.num_anom_outlier:]) # get indices of samples of this class with highest entropy 
        cls_low_ent_idx = list(cls_idx_ent_sorted[:args.num_anom_inlier]) # get indices of samples of this class with lowest entropy

        # cls_he_feas_idx = all_pt_feas[cls_high_ent_idx] # get PT feature set of samples of this class with high entropy
        # cls_le_feas_idx = all_pt_feas[cls_low_ent_idx] # get PT feature set of samples of this class with low entropy
        if start == True:
            known_idx = cls_low_ent_idx
            ood_idx = high_ent_idx
            start = False
        else:
            known_idx = np.concatenate((known_idx, cls_low_ent_idx))
            ood_idx = np.concatenate((ood_idx, high_ent_idx))
        
    
    print("Known IDX Shape: ", known_idx.shape)
    print("ood_idx shape", ood_idx.shape)

    # Build anomaly detection normal set from pt features of inputs where source model is
    # relatively certain of its normalcy. (Train anomaly detection on this subset. )
    L2_ds = all_pt_feas[known_idx].numpy()
    index = faiss.IndexFlatL2(L2_ds.shape[1])
    index.add(L2_ds)    

    # pseudo label AD ground truth (sort of validation set to choose threshold)
    known_gt = np.zeros_like(known_idx)
    ood_gt = np.ones_like(ood_idx)
    ad_gt = np.concatenate((known_gt, ood_gt))

    # build AD validation set using known and unknown idx. 
    val_L2_ds = all_pt_feas[known_idx].numpy()
    val_L2_ds = np.concatenate((val_L2_ds, all_pt_feas[ood_idx].numpy()))

    # Find optimal theshold using val set
    D, _ = index.search(val_L2_ds, 1)
    ano_score = np.sum(D, axis=1)
    #norm_ano_score = (ano_score - np.amin(ano_score)) / (np.amax(ano_score)-np.amin(ano_score))

    auc_no_thresh = roc_auc_score(ad_gt, ano_score)

    print("Pseudo Label AD AUC: ", auc_no_thresh)

    fpr, tpr, threshold = roc_curve(ad_gt, ano_score)
    optimal_threshold = cutoff_youdens_j(fpr,tpr, threshold)

    threshold_detection = ano_score.copy()
    threshold_detection[ano_score > optimal_threshold] = 1
    threshold_detection[ano_score <= optimal_threshold] = 0

    acc = accuracy_score(ad_gt, threshold_detection)

    print("Pseudo Label AD accuracy: ", acc, "using threshold: ", optimal_threshold)
    
    # Test on entire target dataset to see how well AD working
    D, _ = index.search(all_pt_feas.numpy(), 1)
    ano_score = np.sum(D, axis=1)

    # Find actual AUC to judge how well anomaly detection working
    real_ad_gt = torch.zeros_like(orig_all_label)
    real_ad_gt[orig_all_label == args.class_num] = 1 # outliers are last index

    real_auc = roc_auc_score(real_ad_gt, ano_score)

    print("Real Label AD AUC: ", real_auc)

    threshold_detection = ano_score.copy()
    threshold_detection[ano_score > optimal_threshold] = 1
    threshold_detection[ano_score <= optimal_threshold] = 0

    acc_real = accuracy_score(real_ad_gt, threshold_detection)

    print("Real Label AD accuracy: ", acc_real, "using same threshold found in val set: ", optimal_threshold)

    return pred_label.astype('int'), index, optimal_threshold

def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label_ood(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()

    from sklearn.cluster import KMeans
    kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
    labels = kmeans.predict(ent.reshape(-1,1))

    idx = np.where(labels==1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1
    known_idx = np.where(kmeans.labels_ != iidx)[0]
    all_fea = all_fea[known_idx,:]
    all_output = all_output[known_idx,:]
    predict = predict[known_idx]
    all_label_idx = all_label[known_idx]
    ENT_THRESHOLD = (kmeans.cluster_centers_).mean()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    guess_label = args.class_num * np.ones(len(all_label), )
    guess_label[known_idx] = pred_label

    acc = np.sum(guess_label == all_label.float().numpy()) / len(all_label_idx)
    log_str = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD, accuracy*100, acc*100)

    return guess_label.astype('int')

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--data_mode', type=str, default='') #3 or 10 percent random images
    parser.add_argument('--tar_range_start', type=int, default=0)
    parser.add_argument('--tar_range_end', type=int, default=25)
    parser.add_argument('--inlier_num', type=int, default=65) 
    parser.add_argument('--num_anom_inlier', type=int, default=5) 
    parser.add_argument('--num_anom_outlier', type=int, default=5) 

    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        print("Name: ", i)
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list' + args.data_mode + '.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list' + args.data_mode + '.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list' + args.data_mode + '.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(args.tar_range_start, args.tar_range_end)]

        if args.dset == 'office':
            if args.da == 'pda':
                args.class_num = 31
                args.src_classes = [i for i in range(31)]
                args.tar_classes = [i for i in range(args.tar_range_start, args.tar_range_end)]
        if args.dset == 'VISDA-C':
            if args.da == 'pda':
                args.class_num = 12
                args.src_classes = [i for i in range(12)]
                args.tar_classes = [i for i in range(args.tar_range_start, args.tar_range_end)]
                
        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.makedirs(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_target(args)