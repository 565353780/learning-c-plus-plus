import torch
from torch.utils import data

import warnings

import numpy as np
import cv2
import time


class createDataset(data.Dataset):
    def __init__(self, image_path, size=[320, 160]):
        warnings.simplefilter("ignore")

        self.width = size[0]
        self.height = size[1]
        self.rng = np.random.RandomState(int(time.time()))
        self.path_list = [image_path]
        self.rng.shuffle(self.path_list)

        self.flags = {'size': size}

        self.img = np.zeros(size, np.uint8)

        self.label_img = np.zeros(size, np.uint8)
        self.ins_img = np.zeros((0, size[0], size[1]), np.uint8)

        self.len = len(self.path_list)
        self.mainpath = image_path

    def next(self, path):

        img_path = path + '.jpg'

        frame = cv2.imread(img_path)

        self.rng = np.random.RandomState(int(time.time()))
        if frame is None:
            print("Failed to read:", img_path, 'start read test_img.jpg')
            frame = cv2.imread('test_img.jpg')

        gamma = self.rng.uniform(0.8, 1.4)
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        # 实现映射用的是Opencv的查表函数
        frame = cv2.LUT(frame, gamma_table)

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame_h, frame_w, _ = frame.shape
        crop_factor_h = self.rng.uniform(0.0, 0.01)
        crop_factor_w = self.rng.uniform(0.0, 0.01)
        h = frame_h - frame_h * crop_factor_h
        w = frame_w - frame_w * crop_factor_w
        x = self.rng.uniform(0, int(frame_w - w))
        y = int(frame_h - h) // 2
        crop = np.array([y, y + h, x, x + w]).astype('int')
        frame = frame[crop[0]:crop[1], crop[2]:crop[3]]

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return frame

    def __getitem__(self, idx):

        self.path_from = self.path_list[idx][:-4]
        self.img = self.next(self.path_from)

        self.img = np.array(np.transpose(self.img, (2, 0, 1)), dtype=np.float32)
        print('item : ', self.img.shape)
        return torch.Tensor(self.img)

    def __len__(self):
        return self.len
