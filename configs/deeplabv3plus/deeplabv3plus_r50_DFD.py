_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/DFD.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_e300.py'
]
model = dict(
    decode_head=dict(num_classes=13), auxiliary_head=dict(num_classes=13))
