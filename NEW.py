from brains18 import BrainS18Dataset
from setting import parse_opts
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time

from test_version2 import test, show_acc_curv
from utils.logger import log
from scipy import ndimage
import os
from train_version2_classification import train
from datasets.COVID19_dataset import COVID19_Dataset
import scipy.io as sio


if __name__ == '__main__':
    A = 0
    C_ok = 0
    non_C_ok = 0
    po_all = []
    m_all = []
    pre_all = []
    for fold in range(1, 11):
        # settting
        pthfile = r'H:\0000_项目5_新冠状肺炎\code_pytorch_3d_resnet\MedicalNet\MedicalNet_pytorch_files\MedicalNet_pytorch_files\pretrain\resnet_50.pth'
        sets = parse_opts()
        if not sets.ci_test:
            sets = parse_opts()
            sets.gpu_id = [0]
            sets.no_cuda = False
            # sets.data_root = './toy_data'
            # sets.img_list = './toy_data/COVID19DATA/TRAIN/test_ci.txt'
            sets.img_list = './toy_data/test_ci.txt'
            sets.data_root = './toy_data'
            sets.pretrain_path = pthfile
            sets.num_workers = 1
            sets.model_depth = 50
            sets.resnet_shortcut = 'B'
            sets.input_D = 50
            sets.input_H = 50
            sets.input_W = 50
            sets.pin_memory = False
            sets.new_layer_names = 'fc'
            sets.n_epochs = 60
            sets.batch_size = 2
            sets.save_intervals = sets.n_epochs*109
            # sets.n_epochs = 2
            # sets.batch_size = 2
            # sets.save_intervals = sets.n_epochs * 4
            sets.save_folder = 'E:/transfer_pytorch'
            sets.save_folder2 = 'F:/transfer_pytorch/validation_model'
        # getting model
        torch.manual_seed(sets.manual_seed)
        model, parameters = generate_model(sets)
        print(model)
        # optimizer
        if sets.ci_test:
            params = [{'params': parameters, 'lr': sets.learning_rate}]
        else:
            params = [
                {'params': parameters['base_parameters'], 'lr': sets.learning_rate},
                {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}
            ]
        optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        # train from resume
        if sets.resume_path:
            if os.path.isfile(sets.resume_path):
                print("=> loading checkpoint '{}'".format(sets.resume_path))
                checkpoint = torch.load(sets.resume_path)
                model.load_state_dict(checkpoint['net_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(sets.resume_path, checkpoint['epoch']))

        # getting data
        sets.phase = 'train'
        if sets.no_cuda:
            sets.pin_memory = False
        else:
            sets.pin_memory = True

        ####### 加载数据####################################
        print('开始第%d' % fold + '折')
        # F:/pytorch_data_nii/10_fold/lung/%d
        # F:/pytorch_data_nii/lung_8_slices/10fold
        path_name_test = 'H:/0000_项目5_新冠状肺炎/实验二/nii用于训练medicalnet/original/%d' % fold + '/test'
        path_name_train = 'H:/0000_项目5_新冠状肺炎/实验二/nii用于训练medicalnet/original/%d' % fold + '/train'
        # path_name_train = 'H:/0000_项目5_新冠状肺炎/实验二/nii用于训练medicalnet/t'
        path_name_val = 'F:\original_change_window_center/1/test'
        print('训练集的路径: ', path_name_test)
        print('测试集的路径: ', path_name_train)
        print('验证集的路径: ', path_name_val)

        # test data tensor
        train_data = COVID19_Dataset(path_name_train, sets)
        data_loader_train = DataLoader(train_data, batch_size=sets.batch_size, shuffle=True,
                                       num_workers=sets.num_workers,
                                       pin_memory=sets.pin_memory)
        # validation data tensor
        validation_data = COVID19_Dataset(path_name_val, sets)
        data_loader_val = DataLoader(validation_data, batch_size=sets.batch_size, shuffle=True,
                                     num_workers=sets.num_workers,
                                     pin_memory=sets.pin_memory)

        train_total_acc, global_validation_acc = train(data_loader_train, data_loader_val, model, optimizer, scheduler,
                                                       total_epochs=sets.n_epochs, save_interval=sets.save_intervals,
                                                       save_folder=sets.save_folder, save_folder2=sets.save_folder2,
                                                       sets=sets)
        # 每训练一个迭代记录的训练准确率个数
        # 每ratio个训练准确率对应一个测试准确率
        ratio = 28
        ratio = int(ratio)
        # show_acc_curv(ratio)
        # test data tensor
        testing_data = COVID19_Dataset(path_name_test, sets)
        data_loader_test = DataLoader(testing_data, batch_size=sets.batch_size, shuffle=False, num_workers=1,
                                      pin_memory=False)
        # settting
        sets.target_type = "normal"
        sets.phase = 'test'
        # getting model
        checkpoint = torch.load(sets.resume_path)
        net, _ = generate_model(sets)
        net.load_state_dict(checkpoint['state_dict'])
        # testing
        test_total_acc, COVID19_ok, non_COVID19_ok, all_po, all_masks, all_pres = test(data_loader_test, net, sets)

        os.remove(sets.resume_path)  # 删除
        print('test!!')
        print(test_total_acc)
        A = A + test_total_acc
        C_ok = C_ok + COVID19_ok
        non_C_ok = non_C_ok + non_COVID19_ok
        for n_po in range(len(all_po)):
            po_all.append(all_po[n_po])
            m_all.append(all_masks[n_po])
            pre_all.append(all_pres[n_po])

    print(A/10)
    print(C_ok)
    print(non_C_ok)
    dataNew1 = 'H://0000_项目5_新冠状肺炎//code_matlab//new_experiment//feature_extract//po_all.mat'
    dataNew2 = 'H://0000_项目5_新冠状肺炎//code_matlab//new_experiment//feature_extract//m_all.mat'
    dataNew3 = 'H://0000_项目5_新冠状肺炎//code_matlab//new_experiment//feature_extract//pre_all.mat'
    sio.savemat(dataNew1, {'po_all': po_all})
    sio.savemat(dataNew2, {'m_all': m_all})
    sio.savemat(dataNew3, {'pre_all': pre_all})
    print('wonderful!!!')
