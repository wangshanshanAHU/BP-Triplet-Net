import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
import network
import loss
import gc
from loss import *
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import torch.nn.functional as F
from pre_process import ImageList
import copy
import numpy as np
import random
import torch.optim as optim
import pseudo_labeling
from data_list_index import ImageList_T, RandomIdentitySampler
from utils import ContinuousDataloader
import torchvision.transforms as transforms


DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def mae_loss(output,label,weight,q=1.0):
    one_hot_label=torch.zeros(output.size()).scatter_(1,label.cpu().view(-1,1),1).to(DEVICE)
    mask=torch.eq(one_hot_label,1.0)
    output=F.softmax(output,dim=1)
    mae=(1-torch.masked_select(output,mask)**q)/q
    # print(q,mae)
    return torch.sum(weight*mae)/(torch.sum(weight)+1e-10)

def image_classification_test(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(DEVICE)
            labels = labels
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def train(config):
    source = config["source"]
    target = config["target"]
    ## set pre-process
    prep_dict = {}
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open('new_list/'+source+'_'+target+'_list.txt').readlines(), \
                                transform=prep_dict["target"],second=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                              transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                      shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
            
    val_tranform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    ###triplet
    source_triplet_dataset = ImageList_T(open(data_config["source"]["list_path"]).readlines(), transform=train_transform)
    source_triplet_loader = DataLoader(source_triplet_dataset, batch_size=train_bs,
                                       sampler=RandomIdentitySampler(source_triplet_dataset.labels, num_instances=4), 
                                       num_workers=4, drop_last=True)
    target_pseudo_dataset = ImageList_T(open(data_config["target"]["list_path"]).readlines(), transform=val_tranform)
    target_pseudo_loader = DataLoader(target_pseudo_dataset, batch_size=1,
                                      shuffle=False, num_workers=1)

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.to(DEVICE)

    ad_net = network.AdversarialNetworkSp(base_network.output_num(), 1024,second=True,radius=config["network"]["params"]["radius"])
    ad_net = ad_net.to(DEVICE)

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net)
        base_network = nn.DataParallel(base_network)

    parameter_classifier=[base_network.get_parameters()[2]]
    parameter_feature=base_network.get_parameters()[0:2]+ad_net.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer_classfier = optimizer_config["type"](parameter_classifier, \
                                         **(optimizer_config["optim_params"]))
    optimizer_feature = optimizer_config["type"](parameter_feature, \
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer_feature.param_groups:
        param_lr.append(param_group["lr"])
    param_lr.append(optimizer_classfier.param_groups[0]["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])

    best_acc = 0.0
    best_model = copy.deepcopy(base_network)

    Cs_memory = torch.zeros(class_num, 256).to(DEVICE)
    Ct_memory = torch.zeros(class_num, 256).to(DEVICE)

    source_triplet_iter = ContinuousDataloader(source_triplet_loader)
    triplet_criterion = TripletLoss(margin=0.3)  # triplet loss
    pseudo_loader = []

    for i in range(config["iterations"]):
        if i % config[ "test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders,base_network)
            temp_model = base_network
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = copy.deepcopy(temp_model)
            log_str = "iter: {:05d}, \t precision: {:.4f},\t best_acc:{:.4f}".format(i, temp_acc, best_acc)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)
        if (i + 1) % config["snapshot_interval"] == 0:
            if not os.path.exists("save/rsda_model"):
                os.makedirs("save/rsda_model")
            torch.save(best_model, 'save/rsda_model/'+source+'_'+target+'.pkl')

        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer_classfier = lr_scheduler(optimizer_classfier, i, **schedule_param)
        optimizer_feature = lr_scheduler(optimizer_feature, i, **schedule_param)

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target,gammas,sigmas = iter_target.next()

        inputs_source, inputs_target, labels_source = inputs_source.to(DEVICE), inputs_target.to(DEVICE), labels_source.to(DEVICE)
        gammas,sigmas=gammas.type(torch.Tensor).to(DEVICE),sigmas.type(torch.Tensor).to(DEVICE)
        weight_c = gammas
        weight_c[weight_c < 0.5] = 0.0
        #tri
        inputs_triplet, labels_triplet, _ = next(source_triplet_iter)
        inputs_triplet = inputs_triplet.to(DEVICE)
        labels_triplet = labels_triplet.to(DEVICE)

        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)

        transfer_loss = loss.DANN(features, ad_net)

        pseu_labels_target = torch.argmax(outputs_target, dim=1)
        loss_sm, Cs_memory, Ct_memory = loss.SM(features_source, features_target, labels_source, pseu_labels_target,
                                                Cs_memory, Ct_memory)
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        classifier_loss_target = mae_loss(outputs_target,labels_target,weight_c)

        fc = copy.deepcopy(base_network.fc)
        for param in fc.parameters():
            param.requires_grad = False
        softmax_tar_out = torch.nn.Softmax(dim=1)(fc(features_target))
        H = torch.mean(loss.Entropy(softmax_tar_out))

        # # BNM
        # softmax_tgt = nn.Softmax(dim=1)(outputs_target)
        # _, s_tgt, _ = torch.svd(softmax_tgt)
        # BNM_loss = -torch.mean(s_tgt)  

        lam=network.calc_coeff(i,max_iter=2000)
        ######################
        if (i) % 100 == 0:
            del pseudo_loader           
            gc.collect()
            train_pseudo_dataset = []
            pseudo_loader = []
            pseudo_predict = []
            print('pseudo.....')
            base_network.eval()
            ad_net.eval()
            # iter_test = ContinuousDataloader(target_pseudo_loader)

            count = 0
            for _, (images_pseudo, label_true, _) in enumerate(target_pseudo_loader):
                images_pseudo = images_pseudo.to(DEVICE).detach()                
                with torch.no_grad():
                    features_pseudo, predict = base_network(images_pseudo)
                    predict = predict.detach()
                    features_pseudo = features_pseudo.detach()
                    pseudo_target = predict.argmax(1)
                    top_prob, _ = torch.topk(nn.Softmax(dim=1)(predict), 1)
                    if ((top_prob.data[0].cpu().numpy()) > 0.95):
                        if pseudo_target.data[0] == label_true.data[0]:
                            count += 1
                        s_tuple = (images_pseudo.cpu().data.squeeze(), pseudo_target.cpu().data.squeeze())
                        pseudo_predict.append(pseudo_target.data[0])
                        train_pseudo_dataset.append(s_tuple)
                        del s_tuple            
            print(len(train_pseudo_dataset))
            print('right labels %d' % (count))
            pseudo_loader = DataLoader(train_pseudo_dataset, batch_size=train_bs,
                                       sampler=RandomIdentitySampler(pseudo_predict, num_instances=4), 
                                       num_workers=4, drop_last=True)
                                                                          
            del train_pseudo_dataset
            del pseudo_predict
            gc.collect()

            base_network.train()
            ad_net.train()
        
        if len(pseudo_loader) - 1 >= 0:
            # print('len of triplet: %d' %(len(dset_loaders['pseudo']) - 1))
            try:
                target_pseudo, labels_pseudo_target = next(iter_pseudo_target)          
            except:
                iter_pseudo_target = ContinuousDataloader(pseudo_loader)
                target_pseudo, labels_pseudo_target = next(iter_pseudo_target)

            target_pseudo, labels_pseudo_target = target_pseudo.to(DEVICE), labels_pseudo_target.to(DEVICE)
            
            triplet = torch.cat((inputs_triplet, target_pseudo), 0)            
            label_tri = torch.cat((labels_triplet, labels_pseudo_target), 0)            
            features_triplet, _ = base_network(triplet)
            tri_loss_pseudo, prec = triplet_criterion(features_triplet, label_tri)
            # tri_loss_pseudo = tri_loss_pseudo.item()
            p = network.calc_tri(float(tri_loss_pseudo.item()))
            total_loss = classifier_loss + classifier_loss_target + 0.0 * loss_sm + transfer_loss + config["tradeoff_ent"] * lam * H + p * tri_loss_pseudo
        else:
            total_loss = classifier_loss+classifier_loss_target+ 0.0*loss_sm +transfer_loss+config["tradeoff_ent"]*lam*H 
        optimizer_classfier.zero_grad()
        optimizer_feature.zero_grad()
        total_loss.backward()
        optimizer_classfier.step()
        optimizer_feature.step()

        print('step:{: d},\t,class_loss:{:.4f},\t,transfer_loss:{:.4f},'
              '\t,class_loss_t:{:.4f}'.format(i, classifier_loss.item(),
                                              transfer_loss.item(),classifier_loss_target.item()))
        Cs_memory.detach_()
        Ct_memory.detach_()
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for RSDA-MSTN')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='7', help="device id to run")
    parser.add_argument('--source', type=str, default='amazon',choices=["amazon", "dslr","webcam"])
    parser.add_argument('--target', type=str, default='dslr', choices=["amazon", "dslr", "webcam"])
    parser.add_argument('--test_interval', type=int, default=50, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--stages', type=int, default=5, help="training stages")
    parser.add_argument('--radius', type=float, default=10.0, help="radius")
    args = parser.parse_args()
    s_dset_path = '/home/zh/data/data_list/office/' + args.source + '.txt' #'../../data/office/' + args.source + '_list.txt'
    t_dset_path = '/home/zh/data/data_list/office/' + args.target + '.txt' #'../../data/office/' + args.target + '_list.txt'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    config = {}
    config["source"] = args.source
    config["target"] = args.target
    config["gpu"] = args.gpu_id
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/rsda"
    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"],args.source+"_"+args.target+ "_log.txt"), "a")

    config["prep"] = {'params':{"resize_size":256, "crop_size":224}}
    config["network"] = {"name":network.ResNetCos, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True,"class_num":31,"radius":10} }
    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }
    config["data"] = {"source":{"list_path":s_dset_path, "batch_size":36}, \
                      "target":{"list_path":t_dset_path, "batch_size":36}, \
                      "test":{"list_path":t_dset_path, "batch_size":72}}
    config["out_file"].flush()
    config["iterations"] = 1001
    config["tradeoff_ent"] = 1.0
    if config["source"] == "amazon" and config["target"] == "dslr":
        seed = 1
    elif config["source"] == "amazon" and config["target"] == "webcam":
        seed = 0
        config["tradeoff_ent"] = 0.1
    elif config["source"] == "dslr" and config["target"] == "amazon":
        seed = 1
    else:
        seed = 1

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    best_acc = 0.0
    for k in range(0,args.stages,1):
        print('time:',k)
        config["out_file"].write('\n---time:'+str(k)+'---\n')
        pseudo_labeling.make_new_list(args.source, args.target, iter_times=k)
        temp_acc=train(config)
        if best_acc<temp_acc:
            best_acc=temp_acc
    print("best_acc:",best_acc)
    config["out_file"].write('\nbest_acc:{:.4f}'.format(best_acc))
