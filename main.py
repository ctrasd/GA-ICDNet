import os
import sys
import argparse
import time
import datetime
import numpy as np

import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import *
from torch.optim import lr_scheduler

from tqdm import tqdm
from losses import OnlineTripletLoss, OnlineContrastiveLoss, ContrastiveLoss, OnlineContrastiveLoss, OnlineSimLoss,TripletLoss
import models
from models import model
import data_manager
from img_loader import ImageDataset,ImageDataset_aug
from samplers import RandomIdentitySampler,RandomIdentitySampler_aug
from utils import AverageMeter, save_checkpoint, Logger
from eval_metrics import evaluate,evaluate_rank

sys.path.append("./")
parser = argparse.ArgumentParser(description="Using GA-ICDNet train gait model with triplet-loss and sim-loss and reconst-loss")
# Dataset
parser.add_argument('--max-epoch', default=500, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="start epochs to run")
parser.add_argument("-j", "--workers", default=4, type=int, help="number of data loading workers(default: 4)")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
#parser.add_argument('--cooperative', default=1, type=int, help='whether the probe set only consists of subject with bags')
parser.add_argument('--cooperative', default=1,type=int)
# optimization options
parser.add_argument("--train_batch", default=350, type=int)
parser.add_argument("--test_batch", default=350, type=int)
parser.add_argument("--lr", '--learning-rate', default=0.0002, type=float)
parser.add_argument("--weight-decay", default=5e-4, type=float)
parser.add_argument("--save-dir", default='./save_dir/', type=str)
# Architecture

# Miscs


args = parser.parse_args()
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

train_f = open(args.save_dir+"/train_loss.txt", "w")
test_f = open(args.save_dir+"/test.txt", "w")
from tensorboardX import SummaryWriter
writer = SummaryWriter(comment="_label_group_mask_ones_ablation_early8_sa3_shift005002_0002_500_01label_default_split_truesim_resbottle_con_coo_"+str(args.cooperative))

cont_iter = 1

def main():
    torch.manual_seed(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    print(args)
    # GPU / CPU
    device = torch.device('cuda')

    print("Initializing dataset")
    dataset = data_manager.init_dataset('../imdb/dataset_GEI','id_list.csv',args.cooperative)

    transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.02)),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    # trainLoader
    trainLoader = DataLoader(
        ImageDataset(dataset.train, sample='random', transform=transform),
        sampler=RandomIdentitySampler(dataset.train, num_instances=2),
        batch_size=args.train_batch, num_workers=args.workers
    )

    # test/val queryLoader
    # test/val galleryLoader
    test_probeLoader = DataLoader(
        ImageDataset(dataset.test_probe, sample='dense', transform=transform_test),
        shuffle=False, batch_size=args.test_batch, drop_last=False
    )

    test_galleryLoader = DataLoader(
        ImageDataset(dataset.test_gallery, sample='dense', transform=transform_test),
        shuffle=False, batch_size=args.test_batch, drop_last=False
    )
    model = models.model.ICDNet_group_mask_mask_early_8().to(device=device)
    #model = models.model.ICDNet_mask()
    #model= nn.DataParallel(model).cuda()
    #model = models.model.icdnet().to(device=device)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion_cont = OnlineContrastiveLoss(margin=3)
    #criterion_trip = OnlineTripletLoss(3)
    criterion_trip=TripletLoss(3)
    criterion_sim = OnlineSimLoss()
    criterion_l2 = nn.MSELoss()
    criterion_label = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5,0.999))
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    scheduler=lr_scheduler.MultiStepLR(optimizer,[140], gamma=0.1, last_epoch=-1)

    #checkpoint = torch.load('./save_group_mask_early8_ones2_0002_sa3_500l2_01label_resbottle_shift002_all190_coo0/ep87.pth.tar')
    #model.load_state_dict(checkpoint['state_dict'])
    start_time = time.time()
    best_rank1 = -np.inf
    #args.max_epoch = 1
    cont_iter = 1
    for epoch in range(args.start_epoch, args.max_epoch):
        print("==> {}/{}".format(epoch + 1, args.max_epoch))
        cont_iter = train(epoch, model, criterion_cont, criterion_trip, criterion_sim, criterion_l2,criterion_label, optimizer, scheduler, trainLoader, device, cont_iter)
        if cont_iter > 250000:
            break
        if True:
            print("=============> Test")
            test_f.write("iter" + str(cont_iter) + '\n')
            rank1,correct_rate = test(model, test_probeLoader, test_galleryLoader, device)
            writer.add_scalar("Test/rank1", rank1, epoch)
            writer.add_scalar("Test/correct", correct_rate, epoch)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
            if is_best:
                state_dict = model.state_dict()
                save_checkpoint({
                    'state_dict': state_dict,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                }, is_best, osp.join(args.save_dir, 'ep' + str(epoch + 1) +'.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(epoch, model, criterion_cont, criterion_trip, criterion_sim, criterion_l2,criterion_label, optimizer,scheduler, trainLoader, device, cont_iter):
    model.train(True)
    losses = AverageMeter()
    cont_losses = AverageMeter()
    trip_losses = AverageMeter()
    sim_losses = AverageMeter()
    label_losses=AverageMeter()
    scheduler.step()
    print('lr:',optimizer.state_dict()['param_groups'][0]['lr'])
    lr=optimizer.state_dict()['param_groups'][0]['lr']
    if lr<=0.0001:
        checkpoint = torch.load('./'+args.save_dir+'/best_model.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
    total=0.0
    total_correct=0.0
    for batch_idx, (imgs, pids, bags_label) in tqdm(enumerate(trainLoader)):
        #imgs, pids, bags_label = imgs.to(device=device, dtype=torch.float), \
        #                         pids.to(device=device, dtype=torch.long), \
        #                         bags_label.to(device=device, dtype=torch.long)
        imgs, pids, bags_label = imgs.cuda(),pids.cuda(),bags_label.cuda()
        imgs=imgs.float()
        pids=pids.long()
        bags_label=bags_label.long()
        # identification
        #if cont_iter == 100000:
        #    scheduler.step()
        optimizer.zero_grad()
        output_ident, output_all, ident_features,output_bag = model(imgs)
        _, predicted = torch.max(output_bag, 1)
        #print('predicted:')
        #print(predicted)
        #print('label:')
        #print(bags_label)
        correct=(predicted == bags_label).sum()
        total+=imgs.shape[0]
        total_correct+=correct
        #print('total_correct: total:',total_correct,total)
        #in the input find the image without bag
        dict_hasBag = dict()
        for i in range(len(pids)):
            if bags_label[i] == 0:
                dict_hasBag[pids[i].data.cpu().item()] = i
        imgs_proj_without_bag = torch.zeros(imgs.shape, device=device)
        for i in range(len(pids)):
            imgs_proj_without_bag[i] = imgs[dict_hasBag[pids[i].data.cpu().item()]]
        # triplet loss
        label_loss=criterion_label(output_bag,bags_label)
        trip_loss,sim_loss = criterion_trip(ident_features, pids)
        #sim_loss = criterion_sim(ident_features, pids)
        cont_loss_withoutBag = criterion_l2(output_ident, imgs_proj_without_bag)
        cont_loss_withBag = criterion_l2(output_all, imgs)
        cont_loss = cont_loss_withBag + cont_loss_withoutBag
        if epoch<10:
            loss = trip_loss + sim_loss*0.1 + cont_loss*500+label_loss*0.05
        else:
            loss = trip_loss + sim_loss*0.1 + cont_loss*500+label_loss*0.05

        loss.backward()
        optimizer.step()

        # loss to tensorboardx
        losses.update(loss.item())
        trip_losses.update(trip_loss.item())
        sim_losses.update(sim_loss.item())
        cont_losses.update(cont_loss.item())
        label_losses.update(label_loss.item())
        writer.add_scalar("Train/Loss", losses.val, cont_iter)
        writer.add_scalar("Train/trip_Loss", trip_losses.val, cont_iter)
        writer.add_scalar("Train/sim_Loss", sim_losses.val, cont_iter)
        writer.add_scalar("Train/cont_Loss", cont_losses.val, cont_iter)
        writer.add_scalar("Train/label_loss", label_losses.val, cont_iter)
        cont_iter += 1

        if (cont_iter + 1) % 50 == 0:
            print("iter {}\t Loss {:.4f} ({:.4f}) "
                  "trip_loss {:.4f} ({:.4f}) "
                  "sim_loss {:.4f} ({:.4f}) "
                  "cont_loss {:.4f} ({:.4f})"
                  "label_loss {:.4f} ({:.4f})"
                  "total_correct_rate ({:.5f})".format(cont_iter
                        ,losses.val, losses.avg,
                        trip_losses.val, trip_losses.avg,
                        sim_losses.val, sim_losses.avg,
                        cont_losses.val, cont_losses.avg,
                        label_losses.val,label_losses.avg,
                        total_correct.float()/total))
            train_f.write("iter {}\t Loss {:.4f} ({:.4f}) "
                          "trip_loss {:.4f} ({:.4f}) "
                          "sim_loss {:.4f} ({:.4f}) "
                          "cont_loss {:.4f} ({:.4f})"
                          "label_loss {:.4f} ({:.4f})"
                          "total_correct_rate ({:.5f})".format(cont_iter
                        ,losses.val, losses.avg,
                        trip_losses.val, trip_losses.avg,
                        sim_losses.val, sim_losses.avg,
                        cont_losses.val, cont_losses.avg,
                        label_losses.val,label_losses.avg,
                        total_correct.float()*1.0/total))
            train_f.write('\n')
    return cont_iter


def test(model, queryLoader, galleryLoader, device, ranks=[1, 5, 10, 20]):
    with torch.no_grad():
        model.train(False)
        model.eval()
        correct = 0.
        total = 0.
        total_correct=0.
        qf, q_pids, q_bags = [], [], []
        for batch_idx, (img, pid, bag) in tqdm(enumerate(queryLoader)):
            #total += 1.0
            img = img.to(device=device, dtype=torch.float)
            bag = bag.to(device=device, dtype=torch.long)
            _, _, features,output_bag = model(img)
            #print(output_bag.shape)
            _, predicted = torch.max(output_bag, 1)
            #print('test_predicted:')
            #print(predicted)
            #print('test_label:')
            #print(bag)
            correct=(predicted == bag).sum()
            total+=img.shape[0]
            total_correct+=correct


            features = features.squeeze(0)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pid)
            q_bags.extend(bag)
        qf = torch.cat([x for x in qf])#torch.stack(qf)
        q_pids = np.asarray(q_pids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_bags = [], [], []
        for batch_idx, (img, pid, bag) in tqdm(enumerate(galleryLoader)):
            #total += 1.0
            img = img.to(device=device, dtype=torch.float)
            bag = bag.to(device=device, dtype=torch.long)
            _, _, features,output_bag = model(img)
            _, predicted = torch.max(output_bag, 1)
            #print('test_predicted:')
            #print(predicted)
            #rint('test_label:')
            #print(bag)
            correct=(predicted == bag).sum()
            total+=img.shape[0]
            total_correct+=correct

            features = features.squeeze(0)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pid)
            g_bags.extend(bag)
        gf =torch.cat([x for x in gf]) #torch.stack(gf)
        g_pids = np.asarray(g_pids)
        qf=qf.squeeze()
        gf=gf.squeeze() # 29102*128
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")

        cmc=evaluate_rank(gf,qf,g_pids,q_pids)

        '''
        m, n = qf.size(0), gf.size(0)

        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()

        cmc, mAP = evaluate(distmat, q_pids, g_pids)

        '''
        
        print("Results ----------")
        #print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
            test_f.write("Rank-{:<3}: {:.1%}\n".format(r, cmc[r - 1]))
        print("\n")
        print(total_correct.float(),total)
        print("correct_rate:",total_correct.float()*1.0/total)
        return cmc[0],total_correct.float()*1.0/total


if __name__ == '__main__':
    main()






















