import torch
import numpy as np
import time
import os
import argparse

import cv2

from LapNet import LAPNet
from collections import OrderedDict
from torch.nn.parameter import Parameter
from create_dataset import createDataset

import onnx

ChangetoONNX = True
TestONNX = False
ShowTimeSpend = False
# TestSpeed = False

image_path = '..jpg'


class LapNetResult:
    def __init__(self, ShowTimeSpend=False):
        self.model = None
        self.INPUT_CHANNELS = 3
        self.OUTPUT_CHANNELS = 2
        self.SIZE = [1024, 512]  # [224, 224]
        self.GPU_IDX = 0
        self.ShowTimeSpend = ShowTimeSpend
        self.model_name = None

        torch.cuda.set_device(self.GPU_IDX)

        self.load_model()


    def state_dict(self, model, destination=None, prefix='', keep_vars=False):
        own_state = model.module if isinstance(model, torch.nn.DataParallel) \
            else model
        if destination is None:
            destination = OrderedDict()
        for name, param in own_state._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in own_state._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf
        for name, module in own_state._modules.items():
            if module is not None:
                self.state_dict(module, destination, prefix + name + '.', keep_vars=keep_vars)
        return destination


    def load_state_dict(self, model, state_dict, strict=True):
        own_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) \
            else model.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


    def load_model(self):
        if self.ShowTimeSpend:
            tstart = time.time()

        # model = SegNet(input_ch=self.INPUT_CHANNELS, output_ch=self.OUTPUT_CHANNELS).cuda()
        self.model = LAPNet(input_ch=self.INPUT_CHANNELS, output_ch=self.OUTPUT_CHANNELS, internal_ch=8)

        current_file_list = os.listdir(os.getcwd())
        current_epoch_num = -1
        for file_name in current_file_list:
            if 'LapNet_chkpt_better_epoch' in file_name:
                temp_epoch_num = int(file_name.split('_')[3].split('h')[1])
                if temp_epoch_num > current_epoch_num:
                    current_epoch_num = temp_epoch_num
        chkpt_filename = os.getcwd() + '/LapNet_chkpt_better_epoch' + str(
            current_epoch_num) + "_GPU" + str(self.GPU_IDX) + ".pth"
        if os.path.isfile(chkpt_filename):
            self.model_name = 'LapNet_chkpt_better_epoch' + str(
            current_epoch_num) + "_GPU" + str(self.GPU_IDX)
            checkpoint = torch.load(chkpt_filename)
            start_epoch = checkpoint['epoch']
            print("Found Checkpoint file", chkpt_filename, ".")

            self.model.load_state_dict(checkpoint['net'])
            self.load_state_dict(self.model, self.state_dict(self.model))

        print("Found", torch.cuda.device_count(), "GPU(s).", "Using GPU(s) form idx:", self.GPU_IDX)

        # self.model = torch.nn.DataParallel(self.model)  # = model.cuda()
        self.model.eval()

        if self.ShowTimeSpend:
            print('Spend time on load model : ', time.time() - tstart)


    def get_result(self, image_path):
        if self.ShowTimeSpend:
            tstart = time.time()

        train_dataset = createDataset(image_path, size=self.SIZE)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=24, pin_memory=True,
                                                       shuffle=False, num_workers=0)

        img = list(enumerate(train_dataloader))[0][1]

        img_tensor = torch.tensor(img)

        if self.ShowTimeSpend:
            print('----Spend time on loadImage : ', int((time.time() - tstart)*1000), ' ms')
            tstart = time.time()

        # Predictions
        sem_pred = self.model(img_tensor)

        print(img_tensor.shape)
        print(sem_pred.shape)

        if ChangetoONNX:
            torch.onnx.export(self.model,
                              img_tensor,
                              self.model_name + '.onnx',
                              export_params=True,
                              opset_version=10,
                              do_constant_folding=True,
                              input_names=['input'],
                              output_names=['output']
                              # dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
                              )

        if TestONNX:
            onnx_model = onnx.load(self.model_name + '.onnx')
            onnx.checker.check_model(onnx_model)

        if self.ShowTimeSpend:
            print('----Spend time on getPredTensor : ', int((time.time() - tstart) * 1000), ' ms')
            tstart = time.time()

        # sem_pred=torch.floor(sem_pred)
        seg_map = torch.squeeze(sem_pred, 0).cpu().detach().numpy()

        seg_show = seg_map[1]
        (h,w)=seg_show.shape

lapnet_result = LapNetResult(ShowTimeSpend)

lapnet_result.get_result(image_path)