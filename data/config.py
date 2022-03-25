# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,  # 梯度消失和梯度爆炸
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 1,   # 批大小
    'ngpu': 1,
    'epoch': 5000,   # 单次epoch的迭代次数减少，提高运行速度。（单次epoch=(全部训练样本/batchsize)/iteration=1
    'decay1': 190,
    'decay2': 220,   # 衰变
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,  # 输入通道
    'out_channel': 64   # 输出通道
}

cfg_re50 = {

    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,  # True
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 300,
    'decay1': 120,
    'decay2': 250,
    'image_size': 840,
    'pretrain': False,  # False
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,  # 256
    'out_channel': 256
}

cfg_re50c = {

    'name': 'Resnet50c',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,  # True
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 300,
    'decay1': 120,
    'decay2': 250,
    'image_size': 840,
    'pretrain': False,  # False
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,  # 256
    'out_channel': 256
}
cfg_re50s = {

    'name': 'Resnet50s',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,  # True
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 300,
    'decay1': 120,
    'decay2': 250,
    'image_size': 840,
    'pretrain': False,  # False
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,  # 256
    'out_channel': 256
}
cfg_re50cs = {

    'name': 'Resnet50cs',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,  # True
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 300,
    'decay1': 120,
    'decay2': 250,
    'image_size': 840,
    'pretrain': False,  # False
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,  # 256
    'out_channel': 256
}
cfg_gnet = {
    'name': 'ghostnet',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,
    'loc_weight': 2.0,
    'gpu_train': False,  #  原为true
    'batch_size': 1,  #  原为16
    'ngpu': 1,
    'epoch': 1,  # 100
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,  # 原为false
    'return_layers': {'blocks1': 1, 'blocks2': 2, 'blocks3': 3},
    # 'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,  # 32
    'out_channel': 64   # 64
}
cfg_mnetv3 = {
    'name': 'mobilev3',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,  # 梯度消失和梯度爆炸
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 190,
    'decay2': 220,
    'image_size': 680,
    'pretrain': False,
    'return_layers': {'bneck1': 1, 'bneck2': 2, 'bneck3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re152 = {

    'name': 'Resnet152',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,  # True
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 2,
    'ngpu': 1,
    'epoch': 120,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,  # False
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,  # 256
    'out_channel': 256
}
cfg_re1520 = {

    'name': 'Resnet1520',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,  # True
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 300,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': False,  #
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,  # 256
    'out_channel': 256
}
