_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/DFD.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_e300.py'
]
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.05)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
