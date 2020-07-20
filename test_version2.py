from setting import parse_opts
from datasets.brains18 import BrainS18Dataset
from model import generate_model
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy import ndimage
import nibabel as nib
import sys
import os

from train_version2_classification import global_train_acc, global_validation_acc, train_Loss_list
from utils.file_process import load_lines
import numpy as np
import matplotlib.pyplot as plt

global_test_acc = []


# train_Loss_list = []
# global_valid_loss = []
# validation_acc = []
# global_train_acc = []
# global_validation_acc = []


def show_acc_curv(ratio):
    # 训练准确率曲线的x、y
    train_x = list(range(len(train_Loss_list)))
    train_y = train_Loss_list
    train_y2 = global_train_acc

    # 验证准确率曲线的x、y
    # 每ratio个训练准确率对应一个测试准确率
    # test_x = train_x[ratio-1::ratio]
    # test_y = global_validation_acc

    plt.title('COCID19&CAP RESNET-50 ACC')

    plt.plot(train_x, train_y, color='green', label='training loss')
    plt.plot(train_x, train_y2, color='red', label='training accuracy')
    # plt.plot(test_x, test_y, color='red', label='validation accuracy')

    # 显示图例
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('accs')

    plt.show()


def test(data_loader, model, sets):
    model.eval()  # for testing
    batches_per_epoch = len(data_loader)
    # 样本批次训练的损失函数值的和
    train_loss = 0
    # 识别正确的样本数
    ok = 0
    COVID19_ok = 0
    non_COVID19_ok = 0
    total_acc = 0
    all_po = []
    all_masks = []
    all_pres = []
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
            probs = torch.nn.functional.softmax(probs, dim=1)
        po, pre = torch.max(probs.data, 1)
        ok += (pre == label_masks).sum()
        po = po.cpu()
        label_masks = label_masks .cpu()
        pre = pre.cpu()
        po = po.numpy()
        label_masks = label_masks.numpy()
        pre = pre.numpy()
        for n_po in range(len(pre)):
            all_po.append(po[n_po])
            all_masks.append(label_masks[n_po])
            all_pres.append(pre[n_po])
        # po = po.cuda()
        # label_masks = label_masks.cuda()
        # pre = pre.cuda()
        po = torch.from_numpy(po)
        pre = torch.from_numpy(pre)
        label_masks = torch.from_numpy(label_masks)
        for n_label in range(len(pre)):
            if pre[n_label] == label_masks[n_label] and pre[n_label] == 1:
                COVID19_ok = 1 + COVID19_ok
            if pre[n_label] == label_masks[n_label] and pre[n_label] == 0:
                non_COVID19_ok = 1 + COVID19_ok
        # 已训练的样本数
        traind_total = (batch_id-1)*sets.batch_size+len(label_masks)
        # 准确度
        acc = 100. * ok / traind_total
        print('batch:{}, ACC:{}\n'.format(batch_id, acc))
        # 记录测试准确率以输出变化曲线
        global_test_acc.append(acc)

    return acc, COVID19_ok, non_COVID19_ok, all_po, all_masks, all_pres
