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
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
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
    txt_tar = open(args.t_dset_path).readlines()

    # determines how many lines to read from dataset (how many classes) as passed in as argument
    if args.data_to_load > 0:
        txt_test = open(args.test_dset_path).readlines()[0:args.data_to_load]
    else:
        txt_test = open(args.test_dset_path).readlines()[args.data_to_load:]
    
    if args.da == 'pda':
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
    # with open(args.test_dset_path) as myfile:
    #     txt_test = [next(myfile) for x in range(594)] # determines how many lines to read (how many classes)


    # if not args.da == 'uda':
    #     label_map_s = {}
    #     for i in range(len(args.src_classes)):
    #         label_map_s[args.src_classes[i]] = i

    #     new_tar = []
    #     for i in range(len(txt_tar)):
    #         rec = txt_tar[i]
    #         reci = rec.strip().split(' ')
    #         if int(reci[1]) in args.tar_classes:
    #             if int(reci[1]) in args.src_classes:
    #                 line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
    #                 new_tar.append(line)
    #             else:
    #                 line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
    #                 new_tar.append(line)
    #     txt_tar = new_tar.copy()
    #     #txt_test = txt_tar.copy()
    args.out_file.write(' '.join(txt_tar) + '\n')
    args.out_file.flush()
    args.out_file.write('text tar length: ' + str(len(txt_tar)) + '\n')
    args.out_file.flush()
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

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


def get_embeddings(data, netF, netB):
    with torch.no_grad():
        inputs = data[0]
        labels = data[1]
        inputs = inputs.cuda()
        embeddings = netB(netF(inputs))
        return embeddings, labels
        '''    
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            embeddings = netB(netF(inputs))
            return embeddings, labels
        '''

def get_batch_embeddings(imgs, labels, netF, netB):
    with torch.no_grad():
        inputs = imgs
        inputs = inputs.cuda()
        embeddings = netB(netF(inputs))
        return embeddings, labels

def visualize_source_latent(args, emb_name):
    dset_loaders = data_load(args)
    dset_loader = dset_loaders['test']
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


    if args.visualize_source:
        modelpath = args.output_dir_src + '/source_F.pt'   
        netF.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir_src + '/source_B.pt'   
        netB.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir_src + '/source_C.pt'    
        netC.load_state_dict(torch.load(modelpath))
    else:
        modelpath = args.output_dir + '/target_F_par_0.3.pt'   
        netF.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir + '/target_B_par_0.3.pt'   
        netB.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir + '/target_C_par_0.3.pt'    
        netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    netF.eval()
    netB.eval()

    max_iter = args.max_epoch
    max_iter = args.max_epoch // 2
    iter_num = 0
    xTensor = None
    yTensor = None
    labelTensor = None
    with torch.no_grad():
        iter_test = iter(dset_loaders['test'])
        if not os.path.exists(args.output_dir + "/"+emb_name+"_c1.npy"):
            for i in range(len(dset_loaders['test'])):
                data = iter_test.next()
                print("batch no: ", i)
                
                if args.dset=='VISDA-C':
                    embeddings, labels = get_embeddings(dset_loaders['test'], netF, netB)
                    mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
                    mem_label = torch.from_numpy(mem_label).cuda()
                    '''
                    acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
                    '''
                else:
                    embeddings, labels = get_embeddings(data, netF, netB)
                    #mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
                    #mem_label = torch.from_numpy(mem_label).cuda()
                    '''
                    acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
                    '''
                if xTensor == None:
                    xTensor = embeddings
                    yTensor = labels
                    #labelTensor = mem_label
                else:
                    xTensor = torch.cat((xTensor, embeddings), dim=0)
                    yTensor = torch.cat((yTensor, labels), dim=0)
                    #labelTensor = torch.cat((labelTensor, mem_label), dim=0)
                #if iter_num == 0 or iter_num == 1:
                # to check that tensor types did not change after conversion
                # print(xTensor.shape)
                # print(yTensor.shape)
                #iter_num += 1
            
            #torch.save(xTensor, 'xTensor_' + emb_name + '.pt')
            #torch.save(yTensor, 'yTensor_' + emb_name + '.pt')
            curr_x = xTensor.cpu()
            curr_y = yTensor.cpu()
            tsne = TSNE(n_components=2, verbose=1, random_state=123)
            z = tsne.fit_transform(curr_x) 
            df = pd.DataFrame()
            df["y"] = curr_y.detach().numpy()
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]
            y = curr_y.detach().numpy()
            c1 = df["comp-1"].to_numpy()
            c2 = df["comp-2"].to_numpy()
            np.save(args.output_dir + "/"+emb_name+"_y.npy", y)
            np.save(args.output_dir + "/"+emb_name+"_c1.npy", c1)
            np.save(args.output_dir + "/"+emb_name+"_c2.npy", c2)
        else:
            print(args.output_dir + "/"+emb_name+"_y.npy")
            y = np.load(args.output_dir + "/"+emb_name+"_y.npy")
            c1 = np.load(args.output_dir + "/"+emb_name+"_c1.npy")
            c2 = np.load(args.output_dir + "/"+emb_name+"_c2.npy")
        
        df = pd.DataFrame()
        # # preprocess numpy
        # first_idx = np.where(y == args.vis_start_cls)[0][0]
        # # reverse array and grab first occurence of end class
        # second_idx = np.where(y == args.vis_end_cls)[0][-1]

        # df["y"] = y[first_idx:second_idx+1]
        # df["comp-1"] = c1[first_idx:second_idx+1]
        # df["comp-2"] = c2[first_idx:second_idx+1]
        if args.data_mode != '_c':
            first_idx = np.where(y == 65)[0][0]
            print(len(y))
            print(first_idx)

            df["y"] = y[:first_idx]
            df["comp-1"] = c1[:first_idx]
            df["comp-2"] = c2[:first_idx]
            
            # sns.scatterplot(x="comp-1", y="comp-2", hue ="y", palette=sns.color_palette("hls", (args.vis_end_cls - args.vis_start_cls) +1), data=df).set(title="SHOT TSNE Visualization")
            ax = sns.scatterplot(x="comp-1", y="comp-2", hue ="y", data=df, palette=sns.color_palette("hls", np.max(y[:first_idx]) - np.min(y[:first_idx])+1), legend=False).set(title="SHOT TSNE Visualization")
            if args.data_to_load <= 0:
                plt.legend(title='Class', fontsize='5', title_fontsize='3', markerscale=.5)

            print(len(y))
            print(first_idx)
            df2 = pd.DataFrame()
            df2["y"] = y[first_idx:]
            df2["comp-1"] = c1[first_idx:]
            df2["comp-2"] = c2[first_idx:]
            print(y[first_idx:])
            # sns.scatterplot(x="comp-1", y="comp-2", hue ="y", palette=sns.color_palette("hls", (args.vis_end_cls - args.vis_start_cls) +1), data=df).set(title="SHOT TSNE Visualization")
            ax = sns.scatterplot(x="comp-1", y="comp-2", hue ="y", data=df2, palette=sns.color_palette("hls",1)).set(title="SHOT TSNE Visualization")
            if args.data_to_load <= 0:
                plt.legend(title='Random Image Class', fontsize='6', title_fontsize='8', markerscale=0.5)
            
            plt.savefig(args.output_dir + "/"+ str(args.data_to_load) + "_" + emb_name + '_tsne_vis.png', dpi=400)
            plt.close()
        else:
            df["y"] = y
            df["comp-1"] = c1
            df["comp-2"] = c2
            
            # sns.scatterplot(x="comp-1", y="comp-2", hue ="y", palette=sns.color_palette("hls", (args.vis_end_cls - args.vis_start_cls) +1), data=df).set(title="SHOT TSNE Visualization")
            ax = sns.scatterplot(x="comp-1", y="comp-2", hue ="y", palette=sns.color_palette("hls", (np.max(y) - np.min(y)) +1), data=df, legend = False).set(title="SHOT TSNE Visualization")
            # if args.data_to_load <= 0:
            #     plt.legend(title='Class', fontsize='2.5', title_fontsize='2', markerscale=0.2)
            
            plt.savefig(args.output_dir + "/"+ str(args.data_to_load) + "_" + emb_name + '_tsne_vis.png', dpi=400)
            plt.close()
    # max_iter = args.max_epoch
    # max_iter = args.max_epoch
    # iter_num = 0
    # xTensor = None
    # yTensor = None

    # for step, data in enumerate(dset_loader): 
    #     print("batch no. ", step)
    #     embeddings, labels = get_batch_embeddings(data[0], data[1], netF, netB)
    #     outputs = netC(embeddings)
    #     if xTensor == None:
    #         #xTensor = embeddings
    #         xTensor = outputs
    #         yTensor = labels
    #     else:
    #         xTensor = torch.cat((xTensor, outputs), dim=0)
    #         yTensor = torch.cat((yTensor, labels), dim=0)

    # curr_x = xTensor.cpu()
    # curr_y = yTensor.cpu()

    # all_output = nn.Softmax(dim=1)(xTensor)
    # _, predict = torch.max(all_output, 1)
    # ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    # ent = ent.reshape(-1,1)
    # ent = ent.float().cpu().detach().numpy()
    # print("ent shape", ent.shape)

    # # tsne = TSNE(n_components=2, verbose=1, random_state=123)
    # # z = tsne.fit_transform(ent) 
    # # df = pd.DataFrame()
    # # df["y"] = curr_y
    # # df["comp-1"] = z[:,0]
    # # df["comp-2"] = z[:,1]

    # # sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
    # #                 palette=sns.color_palette("hls", torch.max(yTensor) - torch.min(yTensor) +1),
    # #                 data=df).set(title="SHOT TSNE Visualization")

    # plt.scatter(curr_y, ent)
    # plt.savefig(args.output_dir + "/"+ str(args.data_to_load) + "_" + emb_name + '_tsne_vis.png', dpi=400)
    # plt.close()

def train_target_2(args, emb_name):
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

    modelpath = args.output_dir + '/target_F_par_0.3.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/target_B_par_0.3.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/target_C_par_0.3.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    netF.eval()
    netB.eval()

    max_iter = args.max_epoch
    max_iter = args.max_epoch // 2
    iter_num = 0
    xTensor = None
    yTensor = None
    labelTensor = None
    with torch.no_grad():
        iter_test = iter(dset_loaders['test'])
        if not os.path.exists(args.output_dir + "/"+emb_name+"_c1.npy"):
            for i in range(len(dset_loaders['test'])):
                data = iter_test.next()
                print("batch no: ", i)
                
                if args.dset=='VISDA-C':
                    embeddings, labels = get_embeddings(dset_loaders['test'], netF, netB)
                    mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
                    mem_label = torch.from_numpy(mem_label).cuda()
                    '''
                    acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
                    '''
                else:
                    embeddings, labels = get_embeddings(data, netF, netB)
                    #mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
                    #mem_label = torch.from_numpy(mem_label).cuda()
                    '''
                    acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
                    '''
                if xTensor == None:
                    xTensor = embeddings
                    yTensor = labels
                    #labelTensor = mem_label
                else:
                    xTensor = torch.cat((xTensor, embeddings), dim=0)
                    yTensor = torch.cat((yTensor, labels), dim=0)
                    #labelTensor = torch.cat((labelTensor, mem_label), dim=0)
                #if iter_num == 0 or iter_num == 1:
                # to check that tensor types did not change after conversion
                # print(xTensor.shape)
                # print(yTensor.shape)
                #iter_num += 1
            
            #torch.save(xTensor, 'xTensor_' + emb_name + '.pt')
            #torch.save(yTensor, 'yTensor_' + emb_name + '.pt')
            curr_x = xTensor.cpu()
            curr_y = yTensor.cpu()
            tsne = TSNE(n_components=2, verbose=1, random_state=123)
            z = tsne.fit_transform(curr_x) 
            df = pd.DataFrame()
            df["y"] = curr_y.detach().numpy()
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]
            y = curr_y.detach().numpy()
            c1 = df["comp-1"].to_numpy()
            c2 = df["comp-2"].to_numpy()
            np.save(args.output_dir + "/"+emb_name+"_y.npy", y)
            np.save(args.output_dir + "/"+emb_name+"_c1.npy", c1)
            np.save(args.output_dir + "/"+emb_name+"_c2.npy", c2)
        else:
            y = np.load(args.output_dir + "/"+emb_name+"_y.npy")
            c1 = np.load(args.output_dir + "/"+emb_name+"_c1.npy")
            c2 = np.load(args.output_dir + "/"+emb_name+"_c2.npy")
        
        df = pd.DataFrame()
        # preprocess numpy
        first_idx = np.where(y == args.vis_start_cls)
        # reverse array and grab first occurence of end class
        second_idx = np.where(y[::-1] == args.vis_end_cls)

        df["y"] = y[first_idx:second_idx]
        df["comp-1"] = c1[first_idx:second_idx]
        df["comp-2"] = c2[first_idx:second_idx]
        
        ax = sns.scatterplot(x="comp-1", y="comp-2", hue ="y", palette=sns.color_palette("hls", (np.maximum(y) - np.minimum(y)) +1), data=df, legend=False).set(title="SHOT TSNE Visualization")
        ax.get_legend.remove()
        # if args.data_to_load <= 0:
        #     plt.legend(title='Class', fontsize='2.5', title_fontsize='2', markerscale=0.2)
            
        plt.savefig(args.output_dir + "/"+ str(args.data_to_load) + "_" + emb_name + '_tsne_vis.png', dpi=400)
        plt.close()

def train_target(args, emb_name):
    dset_loaders = data_load(args)
    dset_loader = dset_loaders['test']
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

    modelpath = args.output_dir + '/target_F_par_0.3.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/target_B_par_0.3.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/target_C_par_0.3.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    netF.eval()
    netB.eval()

    max_iter = args.max_epoch
    max_iter = args.max_epoch
    iter_num = 0
    xTensor = None
    yTensor = None
    #labelTensor = None

    # while iter_num < max_iter:
    #     print("Epoch", iter_num)
        
    #     if args.dset=='VISDA-C':
    #         embeddings, labels = get_embeddings(dset_loaders['test'], netF, netB)
    #         mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
    #         mem_label = torch.from_numpy(mem_label).cuda()
    #         '''
    #         acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
    #         log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
    #         '''
    #     else:
    #         embeddings, labels = get_embeddings(dset_loaders['test'], netF, netB)
    #         mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
    #         mem_label = torch.from_numpy(mem_label).cuda()
    #         '''
    #         acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
    #         log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
    #         '''
    #     if xTensor == None:
    #         xTensor = embeddings
    #         yTensor = labels
    #         labelTensor = mem_label
    #     else:
    #         xTensor = torch.cat((xTensor, embeddings), dim=0)
    #         yTensor = torch.cat((yTensor, labels), dim=0)
    #         labelTensor = torch.cat((labelTensor, mem_label), dim=0)
    #     #if iter_num == 0 or iter_num == 1:
    #     # to check that tensor types did not change after conversion
    #     # print(xTensor.shape)
    #     # print(yTensor.shape)
    #     iter_num += 1
    
    #torch.save(xTensor, 'xTensor_' + emb_name + '.pt')
    #torch.save(yTensor, 'yTensor_' + emb_name + '.pt')
    for step, data in enumerate(dset_loader): 
        print("batch no. ", step)
        embeddings, labels = get_batch_embeddings(data[0], data[1], netF, netB)
        if xTensor == None:
            xTensor = embeddings
            yTensor = labels
        else:
            xTensor = torch.cat((xTensor, embeddings), dim=0)
            yTensor = torch.cat((yTensor, labels), dim=0)

    curr_x = xTensor.cpu()
    curr_y = yTensor.cpu()
    # curr_labels = labelTensor.cpu()
    # if args.compare_pred_labels != 0:
    #     print("y and pred label shape", curr_y.shape, curr_labels.shape)
    #     non_random_idx = (curr_y[:] != 65).nonzero().squeeze(1)
    #     comp_y = curr_y[non_random_idx]
    #     comp_labels = curr_labels[non_random_idx]
    #     print("y and pred label shape", comp_y.shape, comp_labels.shape)
    #     label_compare = torch.eq(comp_y, comp_labels).numpy()
    #     correct_count = np.count_nonzero(label_compare)
    #     correct_ratio = correct_count / label_compare.size 
    #     print("CORRECT PREDICTION RATIO OF NON RANDOM IMAGES*******", correct_ratio)

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(curr_x) 
    df = pd.DataFrame()
    df["y"] = curr_y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", torch.max(yTensor) - torch.min(yTensor) +1),
                    data=df).set(title="SHOT TSNE Visualization")
    plt.savefig(args.output_dir + "/"+ str(args.data_to_load) + "_" + emb_name + '_tsne_vis.png', dpi=400)
    plt.close()


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=1, help="number of workers")
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
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--data_to_load', type=int, default=594)
    parser.add_argument('--visualize_source', type=int, default=0)
    parser.add_argument('--data_mode', type=str, default='', choices=['_c','_3p', '_10p', '_3p_cl', '_10p_cl']) #3 or 10 percent random images
    parser.add_argument('--tar_range_start', type=int, default=0)
    parser.add_argument('--tar_range_end', type=int, default=25)
    parser.add_argument('--inlier_num', type=int, default=65)
    parser.add_argument('--vis_start_cls', type=int, default=0)
    parser.add_argument('--vis_end_cls', type=int, default=65)
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
        if i != 3:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list' + args.data_mode + '.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list' + args.data_mode + '.txt'
        args.test_dset_path = args.t_dset_path

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(args.tar_range_start, args.tar_range_end)]
            elif args.da == 'oda':
                args.class_num = args.inlier_num
        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.makedirs(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'vislog_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()

        if args.visualize_source:
            visualize_source_latent(args, "source_only_" + "_" + str(args.vis_start_cls)+"_"+str(args.vis_end_cls) + "_" + args.data_mode + " " +names[args.s][0].upper()+names[args.t][0].upper())
        else:
            visualize_source_latent(args, str(args.vis_start_cls)+"_"+str(args.vis_end_cls) + "_" + args.data_mode +" "  + names[args.s][0].upper()+names[args.t][0].upper())