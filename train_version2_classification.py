'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''
from COVID19_dataset import COVID19_Dataset
from setting import parse_opts
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time

from utils.logger import log
from scipy import ndimage
import os

train_Loss_list = []
global_valid_loss = []
validation_acc = []
global_train_acc = []
global_validation_acc = []


def train(data_loader, data_loader_val, model, optimizer, scheduler, total_epochs, save_interval, save_folder,
          save_folder2, sets):
    # settings
    global validation_acc
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    # 原来的代码： loss_seg = nn.CrossEntropyLoss(ignore_index=-1)
    # 定义损失函数
    loss_seg = nn.CrossEntropyLoss()
    print("Current setting is:")
    print(sets)
    print("\n\n")
    if not sets.no_cuda:
        loss_seg = loss_seg.cuda()

    model.train()  # 启用 BatchNormalization 和 Dropout
    train_time_sp = time.time()
    acc = 0
    epoch_acc = 0
    for epoch in range(1, total_epochs + 1):
        log.info('Start epoch {}'.format(epoch))

        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))

        # 样本批次训练的损失函数值的和
        train_loss = 0
        # 识别正确的样本数
        ok = 0
        batch_acc = 0
        # 0: 数据， 1：标签， 2：名字
        for batch_id, (volumes, label_masks) in enumerate(data_loader):
            # volumes, target = volumes.to(torch.device), label_masks.to(torch.device)
            # list 2 tensor
            # print(volumes.shape)
            # getting data batch
            batch_id = batch_id + 1
            batch_id_sp = epoch * batch_id

            if not sets.no_cuda:
                volumes = volumes.cuda()

            optimizer.zero_grad()
            out_masks = model(volumes)

            # calculating loss
            if not sets.no_cuda:
                label_masks = label_masks.cuda()

            loss_value_seg = loss_seg(out_masks, label_masks.long())
            loss = loss_value_seg
            loss.backward()
            optimizer.step()

            # 已训练的样本数
            traind_total = (batch_id - 1) * sets.batch_size + len(label_masks)
            # 累加损失值和训练样本数
            train_loss += loss.item()
            train_Loss_list.append(train_loss/traind_total)
            _, predicted = torch.max(out_masks.data, 1)
            # 累加识别正确的样本数
            ok += (predicted == label_masks).sum()

            # 准确度
            acc = 100. * ok / traind_total
            # 记录训练准确率以输出变化曲线
            global_train_acc.append(acc)
            batch_acc = acc
            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_seg.item(), avg_batch_time))

            if not sets.ci_test:
                # save model
                t = batch_id_sp % save_interval
                # if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)

                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
                    torch.save({
                        'ecpoch': epoch,
                        'batch_id': batch_id,
                        'state_dict': model.state_dict(),
                        'optimeizer': optimizer.state_dict()},
                        model_save_path)

                    sets.resume_path = model_save_path

                # if batch_id != 0 and batch_id % batches_per_epoch == 0:
                #     model_save_path2 = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder2, epoch, batch_id)
                #     model_save_dir2 = os.path.dirname(model_save_path2)
                #     if not os.path.exists(model_save_dir2):
                #         os.makedirs(model_save_dir2)
                #     torch.save({
                #         'ecpoch': epoch,
                #         'batch_id': batch_id,
                #         'state_dict': model.state_dict(),
                #         'optimeizer': optimizer.state_dict()},
                #         model_save_path2)
                #     sets.resume_path2 = model_save_path2
        print(batch_acc)
        epoch_acc = epoch_acc + batch_acc
        # 每一个 epoch 验证一次
        # settting
        # sets.target_type = "normal"
        # sets.phase = 'test'
        # # getting model
        # checkpoint = torch.load(sets.resume_path2)
        # net, _ = generate_model(sets)
        # net.load_state_dict(checkpoint['state_dict'])
        # # testing
        # validation1(data_loader_val, net, sets)
        # os.remove(sets.resume_path2)  # 删除
    print('Finished training')
    print(epoch_acc / total_epochs)
    if sets.ci_test:
        exit()

    return epoch_acc / total_epochs, global_validation_acc


def validation1(data_loader, model, sets):
    model.eval()  # for testing
    # 定义损失函数
    loss_seg = nn.CrossEntropyLoss()
    batches_per_epoch = 2 * len(data_loader) / sets.batch_size
    # 样本批次训练的损失函数值的和
    val_loss = 0
    # 识别正确的样本数
    ok = 0
    total_acc = 0
    for batch_id, (batch_data, label_masks) in enumerate(data_loader):
        batch_id = batch_id + 1
        # forward
        volume = batch_data
        if not sets.no_cuda:
            volume = volume.cuda()
        if not sets.no_cuda:
            label_masks = label_masks.cuda()
        with torch.no_grad():
            probs = model(volume)

        loss_value_seg = loss_seg(probs, label_masks.long())
        loss = loss_value_seg
        val_loss += loss.item()
        _, pre = torch.max(probs.data, 1)
        ok += (pre == label_masks).sum()
        # 已训练的样本数
        traind_total = (batch_id-1)*sets.batch_size+len(label_masks)
        # 准确度
        acc = 100. * ok / traind_total
        print('batch:{}, ACC:{}\n'.format(batch_id, acc))
        # 记录测试准确率以输出变化曲线
        total_acc = acc
    global_valid_loss.append(val_loss / batches_per_epoch)
    global_validation_acc.append(total_acc)
    return
