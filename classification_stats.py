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
from torchvision.utils import save_image
import pandas as pd
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
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    args.out_file.write("length of target domain: " + str(len(txt_tar)) + '\n')
    args.out_file.flush()
    print("length of target domain", len(txt_tar))

    if args.da == 'oda' or args.da == 'pda':
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
    # elif args.da == 'pda':
    #     label_map_s = {}
    #     for i in range(len(args.src_classes)):
    #         label_map_s[args.src_classes[i]] = i

    #     new_tar = []
    #     for i in range(len(txt_tar)):
    #         rec = txt_tar[i]
    #         reci = rec.strip().split(' ')
    #         if int(reci[1]) in args.tar_classes or int(reci[1]) == len(args.src_classes):
    #             if int(reci[1]) in args.src_classes:
    #                 line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
    #                 new_tar.append(line)
    #             else:
    #                 line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
    #                 new_tar.append(line)
    #     txt_tar = new_tar.copy()
    #     txt_test = txt_tar.copy()
    args.out_file.write(' '.join(txt_tar) + '\n')
    args.out_file.flush()
    args.out_file.write('text tar length: ' + str(len(txt_tar)) + '\n')
    args.out_file.flush()
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(args, loader, netF, netB, netC, flag=False, threshold=0.1):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        if args.da == 'oda':
            # added OOD part
            all_output = nn.Softmax(dim=1)(all_output)
            print("got to the entropy part")
            ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)

            from sklearn.cluster import KMeans
            kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
            labels = kmeans.predict(ent.reshape(-1,1))

            idx = np.where(labels==1)[0]
            iidx = 0
            if ent[idx].mean() > ent.mean():
                iidx = 1
            predict[np.where(labels==iidx)[0]] = args.class_num
            #predict[np.where(all_label<40)[0]] = args.class_num
            print("all label shape", all_label.shape)
            print("all label min", torch.min(all_label))
            print("all label max", torch.max(all_label))
            print("predict shape", predict.shape)
            print("pred min", torch.min(predict))
            print("pred max", torch.max(predict))
            # matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
            # print("matrix shape", matrix.shape)
            # font = {'size'   : 0}
            # plt.rc('font', **font)
            # disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
            # disp.plot()
            # plt.savefig(args.output_dir + "/"+ str(args.data_mode) + "_confusion_matrix.png", dpi=400)

            matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
            print("matrix shape", matrix.shape)
            font = {
                'size'   : 4,
                # 'labelsize'  : 7,
                # 'titlesize' : 7
                }
            plt.rc('font', **font)
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
            #print(disp.text_)
            plot = disp.plot(include_values=True)
            plot.ax_.xaxis.label.set_size(5)
            plot.ax_.yaxis.label.set_size(5)
            #disp.text_ = None
            plt.savefig(args.output_dir + "/"+ str(args.data_mode) + "_confusion_matrix.png", dpi=400)

            plt.close()
            # matrix = matrix[np.unique(all_label).astype(int),:]
            # if matrix.shape[0] != matrix.shape[1]:
            #     newrow = np.ones(matrix.shape[1])
            #     matrix = np.vstack([matrix, newrow])
            df = pd.DataFrame(get_tpr_fnr_fpr_tnr(matrix)).transpose()
            avg_tpr = df["TPR"].mean()
            avg_fnr = df["FNR"].mean()
            avg_fpr = df["FPR"].mean()
            avg_tnr = df["TNR"].mean()
            avg_str = "Average TPR: " + str(avg_tpr) + " Average FNR: " + str(avg_fnr) + " Average FPR: " + str(avg_fpr) + " Average TNR: " + str(avg_tnr)
            log_str = df.to_string()
            print(log_str)
            print(avg_str)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            args.out_file.write(avg_str + '\n')
            args.out_file.flush()
        else:
            all_output = nn.Softmax(dim=1)(all_output)
            # ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)

            # from sklearn.cluster import KMeans
            # kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
            # labels = kmeans.predict(ent.reshape(-1,1))

            # idx = np.where(labels==1)[0]
            # iidx = 0
            # if ent[idx].mean() > ent.mean():
            #     iidx = 1
            # predict[np.where(labels==iidx)[0]] = args.class_num

            pos_labels = np.arange(0,args.class_num+1)
            matrix = confusion_matrix(all_label, torch.squeeze(predict).float(), labels = pos_labels)
            print("matrix shape", matrix.shape)
            font = {
                'size'   : 3,
                # 'labelsize'  : 7,
                # 'titlesize' : 7
                }
            plt.rc('font', **font)
            plt.xlabel
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
            #print(disp.text_)
            plot = disp.plot(include_values=False)
            plot.ax_.xaxis.label.set_size(5)
            plot.ax_.yaxis.label.set_size(5)
            #disp.text_ = None
            plt.savefig(args.output_dir + "/"+ str(args.data_mode) + "_confusion_matrix.png", dpi=400)
            plt.close()
            # matrix = matrix[np.unique(all_label).astype(int),:]
            # if matrix.shape[0] != matrix.shape[1]:
            #     newrow = np.ones(matrix.shape[1])
            #     matrix = np.vstack([matrix, newrow])
            # df = pd.DataFrame(get_tpr_fnr_fpr_tnr(matrix)).transpose()
            # avg_tpr = df["TPR"].mean()
            # avg_fnr = df["FNR"].mean()
            # avg_fpr = df["FPR"].mean()
            # avg_tnr = df["TNR"].mean()
            # avg_str = "Average TPR: " + str(avg_tpr) + " Average FNR: " + str(avg_fnr) + " Average FPR: " + str(avg_fpr) + " Average TNR: " + str(avg_tnr)
            # log_str = df.to_string()
            # print(log_str)
            # print(avg_str)
            # args.out_file.write(log_str + '\n')
            # args.out_file.flush()
            # args.out_file.write(avg_str + '\n')
            # args.out_file.flush()

        # print("matrix shape", matrix.shape)
        # print("matrix")
        # print(matrix)

        # FP = matrix.sum(axis=0) - np.diag(matrix)  
        # FN = matrix.sum(axis=1) - np.diag(matrix)
        # TP = np.diag(matrix)
        # TN = matrix.sum() - (FP + FN + TP)

        # FP = FP.astype(float)
        # FN = FN.astype(float)
        # TP = TP.astype(float)
        # TN = TN.astype(float)


        # # Sensitivity, hit rate, recall, or true positive rate
        # TPR = TP/(TP+FN)
        # # Specificity or true negative rate
        # TNR = TN/(TN+FP) 
        # # Precision or positive predictive value
        # PPV = TP/(TP+FP)
        # # Negative predictive value
        # NPV = TN/(TN+FN)
        # # Fall out or false positive rate
        # FPR = FP/(FP+TN)
        # # False negative rate
        # FNR = FN/(TP+FN)
        # # False discovery rate
        # FDR = FP/(TP+FP)

        # # Overall accuracy
        # ACC = (TP+TN)/(TP+FP+FN+TN)

        # #acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        # #unknown_acc = acc[-1:].item()
        # log_str = "TPR: " + str(TPR) + " TNR: " + str(TNR) + " PPV: " + str(PPV) + " NPV: " + str(NPV) + " FPR: " + str(FPR) + " FNR: " + str(FNR) + " FDR: " + str(FDR) + " ACC: " + str(ACC)
        # args.out_file.write(log_str + '\n')
        # args.out_file.flush()
        # print(log_str)

def get_tpr_fnr_fpr_tnr(cm):
    """
    This function returns class-wise TPR, FNR, FPR & TNR
    [[cm]]: a 2-D array of a multiclass confusion matrix
            where horizontal axes represent actual classes
            and vertical axes represent predicted classes
    {output}: a dictionary of class-wise accuracy parameters
    """
    dict_metric = dict()
    n = len(cm[0])
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    array_sum = sum(sum(cm))
    #initialize a blank nested dictionary
    for i in range(1, n+1):
        keys = str(i)
        dict_metric[keys] = {"TPR":0, "FNR":0, "FPR":0, "TNR":0}
    # calculate and store class-wise TPR, FNR, FPR, TNR
    for i in range(n):
        for j in range(n):
            if i == j:
                keys = str(i+1)
                tp = cm[i, j]
                fn = row_sums[i] - cm[i, j]
                dict_metric[keys]["TPR"] = tp / (tp + fn)
                dict_metric[keys]["FNR"] = fn / (tp + fn)
                fp = col_sums[i] - cm[i, j]
                tn = array_sum - tp - fn - fp
                dict_metric[keys]["FPR"] = fp / (fp + tn)
                dict_metric[keys]["TNR"] = tn / (fp + tn)
    return dict_metric

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def get_stats(args):
    dset_loaders = data_load(args)
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

    # load pretrained networks
    args.modelpath = osp.join(args.output_dir, "target_F_" + args.savename + ".pt")   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = osp.join(args.output_dir, "target_B_" + args.savename + ".pt")  
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = osp.join(args.output_dir, "target_C_" + args.savename + ".pt")    
    netC.load_state_dict(torch.load(args.modelpath))
    for k, v in netC.named_parameters():
        v.requires_grad = False
    for k, v in netB.named_parameters():
        v.requires_grad = False
    for k, v in netF.named_parameters():
        v.requires_grad = False
    
    netC.eval()
    netF.eval()
    netB.eval()
    #mem_label, ENT_THRESHOLD = obtain_label(dset_loaders['test'], netF, netB, netC, args)
    #mem_label = torch.from_numpy(mem_label).cuda()
    cal_acc(args, dset_loaders['test'], netF, netB, netC, True)            

    return netF, netB, netC

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

    return guess_label.astype('int'), ENT_THRESHOLD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['office-home'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
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
    parser.add_argument('--da', type=str, default='oda', choices=['oda', 'uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--data_mode', type=str, default='') #3 or 10 percent random images
    parser.add_argument('--use_graph', type=bool, default=False) # use graph selection or not
    parser.add_argument('--tar_range_start', type=int, default=0)
    parser.add_argument('--tar_range_end', type=int, default=25)
    args = parser.parse_args()
       
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True
    count = 0
    for i in range(len(names)):
        if i == args.s:
            continue
        elif i != 3:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list' + args.data_mode + '.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list' + args.data_mode + '.txt'
        args.test_dset_path = args.t_dset_path

        if args.dset == 'office-home':
            if args.da == 'oda':
                args.class_num = 25
                args.src_classes = [i for i in range(25)]
                if args.data_mode == '_c':
                    args.tar_classes = [i for i in range(65)]
                    print("clean data mode 65 target classes")
                else:
                    args.tar_classes = [i for i in range(66)]
            elif args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(args.tar_range_start, args.tar_range_end)]
            elif args.da == 'uda':
                args.class_num = 65

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        args.out_file = open(osp.join(args.output_dir, 'statslog_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        
        # args.ae = autoencoder()
        # if not os.path.isfile("./ckps/AE/ae.pt"):
        #     print("pretrained AE not found")
        #     args.ae = train_ae(args)
        #     torch.save(args.ae.state_dict(), "./ckps/AE/ae.pt")
        # else:
        #     print("loaded AE")
        #     args.ae.load_state_dict(torch.load("./ckps/AE/ae.pt"))
        # test_anomaly_detection(args)

        get_stats(args)
        count +=1