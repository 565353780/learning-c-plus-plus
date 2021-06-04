import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# BGR Image Channels Expantion.
class InitialBlock(nn.Module):
    def __init__(self,input_ch,output_ch,bias=False):
        super(InitialBlock, self).__init__()
        # Shortcut
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_ch,
                out_channels=output_ch - 3,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias),
            nn.BatchNorm2d(output_ch - 3),
            nn.PReLU()
        )
        self.pool_max = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        main = self.conv(x)
        max = self.pool_max(x)
        out = torch.cat((main,max), dim=1) # N, C, H, W
        # print(out.shape)
        return out

# Emulating AdaptiveAvgPool2d(Global Avg Pooling 2D)
class CompatibleGlobalAvgPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        self.sz = sz
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(sz), ceil_mode=False)

    def forward(self, x): 
        # inp_size=self.sz
        # if inp_size is None:
        #     inp_size = (x.size()[2],x.size()[3])

        return self.avg_pool2d(x)

#LAP Block
class RegularLAP(nn.Module):
    def __init__(self, input_ch, regularizer_prob,LAP_pooling_shape = None):
        super(RegularLAP, self).__init__()

        LAP_ch = input_ch // 4

        # Channel-wize Attention
        # self.LAP_recognition = nn.Sequential(
        #     CompatibleGlobalAvgPool2d(sz=LAP_pooling_shape),
        #     nn.Conv2d(in_channels = input_ch,
        #         out_channels = LAP_ch,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         bias = False),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels = LAP_ch,
        #         out_channels = LAP_ch // 4,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         bias = False),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels = LAP_ch // 4,
        #         out_channels = LAP_ch,
        #         groups = 1,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         bias = False),
        #     nn.Sigmoid()
        # )

        # Compress Channels
        self.LAP_squeeze_1x1 = nn.Sequential(
            nn.Conv2d(in_channels = input_ch,
                out_channels = LAP_ch,
                groups = 1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias = False),
            nn.BatchNorm2d(LAP_ch)
        )

        # parallel conv x4
        self.LAP_conv_branch_1 = nn.Sequential(
            nn.Conv2d(in_channels = LAP_ch,
                out_channels = LAP_ch,
                groups = LAP_ch,
                kernel_size=(3,1),
                stride=1,
                padding=(1,0),
                dilation=1,
                bias = False),
            nn.BatchNorm2d(LAP_ch),
            
            nn.Conv2d(in_channels = LAP_ch,
                out_channels = LAP_ch,
                groups = LAP_ch,
                kernel_size=(1,3),
                stride=1,
                padding=(0,1),
                dilation=1,
                bias = False),
            nn.BatchNorm2d(LAP_ch)
        )
        self.LAP_conv_branch_2 = nn.Sequential(
            nn.Conv2d(in_channels = LAP_ch,
                out_channels = LAP_ch,
                groups = LAP_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                bias = False),
            nn.BatchNorm2d(LAP_ch)
        )
        self.LAP_conv_branch_3 = nn.Sequential(
            nn.Conv2d(in_channels = LAP_ch,
                out_channels = LAP_ch,
                groups = LAP_ch,
                kernel_size=(1,3),
                stride=1,
                padding=(0,1),
                dilation=1,
                bias = False),
            nn.BatchNorm2d(LAP_ch),
            
            nn.Conv2d(in_channels = LAP_ch,
                out_channels = LAP_ch,
                groups = LAP_ch,
                kernel_size=(3,1),
                stride=1,
                padding=(1,0),
                dilation=1,
                bias = False),
            nn.BatchNorm2d(LAP_ch)
        )
        self.LAP_conv_branch_4 = nn.Sequential(
            nn.Conv2d(in_channels = LAP_ch,
                out_channels = LAP_ch,
                groups = LAP_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias = False),
            nn.BatchNorm2d(LAP_ch)
        )

        self.LAP_group_fusion = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels = input_ch,
                out_channels = input_ch,
                groups = LAP_ch,
                kernel_size=1,
                stride=1,
                padding=0)
        )

        # self.regularizer = nn.Dropout2d(p=regularizer_prob)

        self.bn = nn.BatchNorm2d(input_ch)
        # self.activ = nn.PReLU()
    def forward(self, x):
        main = x
        
        x = self.LAP_squeeze_1x1(x)

        # Multiscale Conv
        a = self.LAP_conv_branch_2(x)
        b = self.LAP_conv_branch_4(x)
        c = self.LAP_conv_branch_1(x)
        d = self.LAP_conv_branch_3(x)
        
        # Reduce kernel disorder.
        x = torch.cat((a,b,c,d), dim=1)
        N,C,H,W = x.size()
        x = self.LAP_group_fusion(x
            .view(N,4,C//4,H,W)
            .permute(0,2,1,3,4).contiguous()
            .view(N,C,H,W))
        
        x = self.bn(x)
        # x = self.regularizer(x)
        
        # x=self.activ(x)
        return main + x

#LAP Block-Dwnsamp
class DownSamplingLAP(nn.Module):
    def __init__(self, input_ch, output_ch,regularizer_prob,with_ind = False,LAP_pooling_shape = None):
        super(DownSamplingLAP, self).__init__()

        self.ret_ind = with_ind
        LAP_ch = output_ch // 4
        stride = 1

        # Channel-wize Attention
        # self.LAP_recognition = nn.Sequential(
        #     CompatibleGlobalAvgPool2d(sz=LAP_pooling_shape),
        #     nn.Conv2d(in_channels = input_ch,
        #         out_channels = input_ch // 4,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         bias = False),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels = input_ch // 4,
        #         out_channels = output_ch,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         bias = False),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels = output_ch,
        #         out_channels = LAP_ch,
        #         groups = 1,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         bias = False),
        #     nn.Sigmoid()
        # )

        # Compress Channels
        self.LAP_squeeze_1x1 = nn.Sequential(
            nn.Conv2d(in_channels = input_ch,
                out_channels = LAP_ch,
                groups = 1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias = False),
            nn.BatchNorm2d(LAP_ch)
        )

        # parallel conv x4
        self.LAP_conv_branch_1 = nn.Sequential(
            nn.Conv2d(in_channels = LAP_ch,
                out_channels = LAP_ch//2,
                groups = LAP_ch//2,
                kernel_size=(3,1),
                stride=1,
                padding=(1,0),
                dilation=1,
                bias = False),
            nn.BatchNorm2d(LAP_ch//2),
            
            nn.Conv2d(in_channels = LAP_ch//2,
                out_channels = LAP_ch//2,
                groups = LAP_ch//2,
                kernel_size=(1,3),
                stride=1,
                padding=(0,1),
                dilation=1,
                bias = False),
            nn.BatchNorm2d(LAP_ch//2)
        )
        self.LAP_conv_branch_2 = nn.Sequential(
            nn.Conv2d(in_channels = LAP_ch,
                out_channels = LAP_ch//2,
                groups = LAP_ch//2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias = False),
            nn.BatchNorm2d(LAP_ch//2)
        )
        self.LAP_conv_branch_3 = nn.Sequential(
            nn.Conv2d(in_channels = LAP_ch,
                out_channels = LAP_ch//2,
                groups = LAP_ch//2,
                kernel_size=(1,3),
                stride=1,
                padding=(0,1),
                dilation=1,
                bias = False),
            nn.BatchNorm2d(LAP_ch//2),
            
            nn.Conv2d(in_channels = LAP_ch//2,
                out_channels = LAP_ch//2,
                groups = LAP_ch//2,
                kernel_size=(3,1),
                stride=1,
                padding=(1,0),
                dilation=1,
                bias = False),
            nn.BatchNorm2d(LAP_ch//2)
        )
        self.LAP_conv_branch_4 = nn.Sequential(
            nn.Conv2d(in_channels = LAP_ch,
                out_channels = LAP_ch//2,
                groups = LAP_ch//2,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias = False),
            nn.BatchNorm2d(LAP_ch//2)
        )

        self.LAP_group_fusion = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels = output_ch,
                out_channels = output_ch,
                groups = LAP_ch,
                kernel_size=1,
                stride=1,
                padding=0)
        )

        # self.regularizer = nn.Dropout2d(p=regularizer_prob)

        self.bn = nn.BatchNorm2d(output_ch)
        # self.activ = nn.PReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=with_ind)

    def forward(self, x):
        if(self.ret_ind):
            x, ind = self.max_pool(x)
        else:
            x = self.max_pool(x)
        main = x

        x = self.LAP_squeeze_1x1(x)
        # Multiscale Conv
        a = self.LAP_conv_branch_2(x)
        b = self.LAP_conv_branch_4(x)
        c = self.LAP_conv_branch_1(x)
        d = self.LAP_conv_branch_3(x)
        # print(np.shape(main))
        # print(np.shape(d))
        # Reduce kernel disorder.
        x = torch.cat((a,b,c,d,main), dim=1)
        N,C,H,W = x.size()
        x = self.LAP_group_fusion(x
            .view(N,4,C//4,H,W)
            .permute(0,2,1,3,4).contiguous()
            .view(N,C,H,W))
        
        # x = self.regularizer(x)
        
        # x=self.activ(x)
        if(self.ret_ind):
            return x, ind
        else:
            return x

#Nearest/Linear Upsmpl       
class UpSampleBottleNeck2(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch,
                 upsample_to,
                 projection_ratio=4,
                 regularizer_prob=0,
                 bias=False):
        super(UpSampleBottleNeck2, self).__init__()

        reduced_depth = input_ch // projection_ratio

        # Shortcut
        self.shortcut_branch = nn.Sequential(
            nn.Conv2d(in_channels=input_ch,
                      out_channels=output_ch,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups = 1,
                      bias=bias),
            nn.BatchNorm2d(output_ch)
        )

        self.conv_branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_ch,
                      out_channels=reduced_depth,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=bias),
            nn.BatchNorm2d(reduced_depth),
            # nn.PReLU()
        )

        self.conv_branch_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=reduced_depth,
                      out_channels=reduced_depth,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      output_padding=1,
                      bias=bias),
            nn.BatchNorm2d(reduced_depth),
            # nn.PReLU()
        )

        self.conv_branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=reduced_depth,
                      out_channels=output_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=bias),
            nn.BatchNorm2d(output_ch),
            # nn.PReLU()
        )

        # self.regularizer = nn.Dropout2d(p=regularizer_prob)
        # self.upsample = nn.functional.interpolate(upsample_to, mode='nearest')
        self.prelu = nn.PReLU()
        self.intr_sz = upsample_to
    def forward(self, x):
        # print("up:", self.intr_sz)
        main = self.shortcut_branch(x)
        ext = self.conv_branch_1(x)
        ext = self.conv_branch_2(ext)
        ext = self.conv_branch_3(ext)
        # ext = self.regularizer(ext)

        out = self.prelu(nn.functional.interpolate(main,self.intr_sz, mode='nearest') + ext)

        return out


class output_up2x(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch,
                 sz):
        super(output_up2x, self).__init__()

        # Shortcut
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels=input_ch,
                      out_channels=output_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(in_channels=output_ch,
                      out_channels=output_ch,
                      groups = 1,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        )

        self.intr_sz = sz
    def forward(self, x):
        x = self.squeeze(x)

        out = nn.functional.interpolate(x,self.intr_sz, mode='nearest')

        return self.smooth(out)

#LAPNet Block Grapth
class LAPNet(nn.Module):
    def __init__(self, input_ch, output_ch, internal_ch, one_channel = False):
        super(LAPNet, self).__init__()
        self.internal_ch = internal_ch
        print("LAPNet internal channel:", self.internal_ch)
        # Initial
        self.initial_block = InitialBlock(input_ch=input_ch, output_ch=self.internal_ch)

        # Encoder
        self.stage1_0 = RegularLAP(input_ch=self.internal_ch, regularizer_prob=0.02*0,LAP_pooling_shape=(256,512))
        self.stage1_D = DownSamplingLAP(input_ch=self.internal_ch, output_ch=self.internal_ch*2, regularizer_prob=0.02*0,LAP_pooling_shape=(128,256))
        self.stage2_0 = RegularLAP(input_ch=self.internal_ch*2, regularizer_prob=0.02*0,LAP_pooling_shape=(128,256))
        self.stage2_1 = RegularLAP(input_ch=self.internal_ch*2, regularizer_prob=0.02*0,LAP_pooling_shape=(128,256))
        self.stage2_2 = RegularLAP(input_ch=self.internal_ch*2, regularizer_prob=0.02*0,LAP_pooling_shape=(128,256))
        self.stage2_3 = RegularLAP(input_ch=self.internal_ch*2, regularizer_prob=0.02*0,LAP_pooling_shape=(128,256))
        self.stage2_D = DownSamplingLAP(input_ch=self.internal_ch*2, output_ch=self.internal_ch*4, regularizer_prob=0.02*0,LAP_pooling_shape=(64,128))
        self.stage3_0 = RegularLAP(input_ch=self.internal_ch*4, regularizer_prob=0.02*0,LAP_pooling_shape=(64,128))
        self.stage3_1 = RegularLAP(input_ch=self.internal_ch*4, regularizer_prob=0.02*0,LAP_pooling_shape=(64,128))
        self.stage3_2 = RegularLAP(input_ch=self.internal_ch*4, regularizer_prob=0.02*0,LAP_pooling_shape=(64,128))
        self.stage3_3 = RegularLAP(input_ch=self.internal_ch*4, regularizer_prob=0.02*0,LAP_pooling_shape=(64,128))
        self.stage3_4 = RegularLAP(input_ch=self.internal_ch*4, regularizer_prob=0.02*0,LAP_pooling_shape=(64,128))

        # Decoder
        self.stage4_0 = RegularLAP(input_ch=self.internal_ch*4, regularizer_prob=0.02*0,LAP_pooling_shape=(64,128))
        self.stage4_1 = RegularLAP(input_ch=self.internal_ch*4, regularizer_prob=0.02*0,LAP_pooling_shape=(64,128))
        self.stage4_2 = RegularLAP(input_ch=self.internal_ch*4, regularizer_prob=0.02*0,LAP_pooling_shape=(64,128))
        self.stage4_3 = RegularLAP(input_ch=self.internal_ch*4, regularizer_prob=0.02*0,LAP_pooling_shape=(64,128))
        self.stage4_4 = RegularLAP(input_ch=self.internal_ch*4, regularizer_prob=0.02*0,LAP_pooling_shape=(64,128))
        self.stage4_5 = UpSampleBottleNeck2(input_ch=self.internal_ch*4, output_ch=self.internal_ch*2, regularizer_prob=0.02*0,upsample_to=(128,256))
        self.stage4_6 = RegularLAP(input_ch=self.internal_ch*2, regularizer_prob=0.02*0,LAP_pooling_shape=(128,256))
        self.stage4_7 = RegularLAP(input_ch=self.internal_ch*2, regularizer_prob=0.02*0,LAP_pooling_shape=(128,256))
        
        self.stage5_0 = UpSampleBottleNeck2(input_ch=self.internal_ch*2, output_ch=self.internal_ch, regularizer_prob=0.02*0,upsample_to=(256,512))
        self.stage5_1 = RegularLAP(input_ch=self.internal_ch, regularizer_prob=0.02*0,LAP_pooling_shape=(256,512))
        self.stage5_2 = RegularLAP(input_ch=self.internal_ch, regularizer_prob=0.02*0,LAP_pooling_shape=(256,512))

        self.sem_out = output_up2x( input_ch=self.internal_ch,
                                    output_ch = output_ch,
                                    sz = (512,1024))

        # self.sem_out = nn.ConvTranspose2d(in_channels=self.internal_ch,
        #                                   out_channels=output_ch,
        #                                   kernel_size=3,
        #                                   stride=2,
        #                                   padding=1,
        #                                   output_padding=1)

        # self.ins_out = nn.ConvTranspose2d(in_channels=64,
        #                                   out_channels=ins_ch,
        #                                   kernel_size=3,
        #                                   stride=2,
        #                                   padding=1,
        #                                   output_padding=1)

        self.prelu = nn.PReLU()
        self.one_channel = one_channel

    def forward(self, x):
        # Initial
        x = self.initial_block(x)
        
        x = self.stage1_0(x)
        x = self.stage1_D(x)
        x = self.stage2_0(x)
        x = self.stage2_1(x)
        x = self.stage2_2(x)
        x = self.stage2_3(x)
        x = self.stage2_D(x)
        x = self.stage3_0(x)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        x = self.stage3_3(x)
        x = self.stage3_4(x)

        x = self.stage4_0(x)
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        x = self.stage4_3(x)
        x = self.stage4_4(x)
        x = self.stage4_5(x)
        x = self.stage4_6(x)
        x = self.stage4_7(x)

        x = self.stage5_0(x)
        
        x = self.stage5_1(x)

        x = self.stage5_2(x)

        x = self.sem_out(x)
        # x = self.prelu(x)

        if(self.one_channel):
           return x[:,1]

        return x

