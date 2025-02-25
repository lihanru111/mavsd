# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class UAVMDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('Background', 'Solidago canadensis', 'Person', 'Road', 'Pine', 'Bush',
               'Reed', 'Stone', 'Cypress ',  'Building ', 'Vegetable garden', 'Electric bicycle')
    PALETTE = [[0, 0, 0], [252, 233, 79], [173, 127, 168], [239, 41, 41], [138, 226, 52],
               [78, 154, 6], [143, 89, 2], [211, 215, 207], [114, 159, 207],
                [233, 185, 110], [196, 160, 0], [164, 0, 0]]

    def __init__(self, split, **kwargs):
        super(UAVMDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
