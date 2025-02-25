# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM


@HEADS.register_module()
class decodehead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,  **kwargs):
        super(decodehead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.layer1 = ConvModule(
            256,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.layer2 = ConvModule(
            512,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.layer3 = ConvModule(
            1024,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.layer4 = ConvModule(
            2048,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.layer = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.con = torch.nn.Conv2d(768, 1024, 1, 1, 0)
        self.bottleneck = ConvModule(
            2048,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module

    def forward(self, inputs1, inputs2):
        """Forward function."""
        cnn_list = []

        for i, input in enumerate(inputs1):
            temp = self.layer[i](input)
            cnn_list.append(resize(
            input=temp,
            size=inputs1[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners))
        output_cnn = torch.cat((cnn_list[0], cnn_list[1], cnn_list[2], cnn_list[3]), dim=1)
        output_trans = self.con(inputs2[3])
        output_trans = resize(
            input=output_trans,
            size=output_cnn.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat((output_cnn, output_trans), dim=1)
        output = self.bottleneck(output)
        output = self.cls_seg(output)
        return output
