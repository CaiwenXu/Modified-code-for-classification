'''
Dataset for training
Written by Whalechen
'''

import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage


class COVID19_Dataset(Dataset):

    def __init__(self, path, sets):
        self.path = path
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase
        self.im_path = []
        self.labels = []
        labels = []
        im_path = []
        im_path, labels = read_path(self.path, self, labels, im_path)
        self.im_path = im_path
        self.labels = labels

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")

        return new_data

    def __len__(self):
        return len(self.im_path)

    def __getitem__(self, idx):

        if self.phase == "train":
            ##################################################
            # read image and labels
            # 更改的部分：full path 是单个nii文件的路径

            # img_name = self.full_path
            #
            # assert os.path.isfile(img_name)
            # img = nibabel.load(img_name)
            # assert img is not None
            #
            # # data processing
            # img_array = self.__training_data_process__(img)
            #
            # # 2 tensor array
            # img_array = self.__nii2tensorarray__(img_array)

            ###### 自己更改的部分#########################################
            # im_path, labels = read_path(self.path, self, labels, im_path)
            # self.im_path = im_path
            # self.labels = labels
            # 处理读入的nii文件
            img = nibabel.load(self.im_path[idx])
            assert img is not None
            img = self.__training_data_process__(img)
            img_array = self.__nii2tensorarray__(img)
            # 标注数据
            label1 = self.labels[idx]
            labels_array = label1
            labels_array = np.array([0 if label1.endswith('0') else 1])
            label = labels_array[0]
            return img_array, label
        # self.len_im_array = len(labels_array)

        elif self.phase == "test":
            img = nibabel.load(self.im_path[idx])
            assert img is not None
            img = self.__training_data_process__(img)
            img_array = self.__nii2tensorarray__(img)
            # 标注数据
            label1 = self.labels[idx]
            labels_array = label1
            labels_array = np.array([0 if label1.endswith('0') else 1])
            label = labels_array[0]
            return img_array, label

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    def __random_center_crop__(self, data, label):
        from random import random
        """
        Random crop
        """
        target_indexs = np.where(label > 0)
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth * 1.0 / 2) * random())
        Y_min = int((min_H - target_height * 1.0 / 2) * random())
        X_min = int((min_W - target_width * 1.0 / 2) * random())

        Z_max = int(img_d - ((img_d - (max_D + target_depth * 1.0 / 2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height * 1.0 / 2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width * 1.0 / 2)) * random()))

        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])

        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)

        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        # pixels = volume[volume > 0]
        pixels = volume
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __training_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()

        # drop out the invalid range
        # data = self.__drop_invalid_range__(data)

        # crop data
        # 人工删除

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data

    def __testing_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data

    # 自己定义的部分


def read_path(path_name, self, labels, im_path):
    for dir_item in os.listdir(path_name):
        # 从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read_path(full_path, self, labels, im_path)
        else:  # 文件
            if dir_item.endswith('.nii'):
                # 开始处理nii文件
                # 调用COVID19_Dataset：
                # read image
                img_name = full_path
                print(img_name)
                assert os.path.isfile(img_name)
                # 处理读入的nii文件
                # img = nibabel.load(img_name)
                # assert img is not None
                # img = self.__training_data_process__(img)
                # listdir的参数是文件夹的路径
                # images.append(img)
                labels.append(path_name)
                im_path.append(img_name)
    return im_path, labels
