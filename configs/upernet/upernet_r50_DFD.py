_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/DFD.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_e300.py'
]
model = dict(
    decode_head=dict(num_classes=13))
