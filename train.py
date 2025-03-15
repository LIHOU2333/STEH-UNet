import argparse
import os
from collections import OrderedDict
from datetime import datetime
from glob import glob
import random
import numpy as np
# Pandas专门用于数据处理和分析
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import yaml
from albumentations import HorizontalFlip, VerticalFlip, Normalize
from albumentations.core.composition import Compose

from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize
import datetime


import losses
from utils.dataset import Dataset

from metrics2 import indicators
from utils.utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter
# #1/4 增加导入SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import shutil
# ROC曲线 AUC面积 增加的库
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')  # 使用无窗口后端
import matplotlib.pyplot as plt
# 直方图均衡化

import cv2


from albumentations.core.transforms_interface import ImageOnlyTransform

# archs 模块通常是一个包含多个模型定义的文件，而 __all__ 是一个特殊的列表，列出了该模块对外公开的所有对象名称。
# # ARCH_NAMES = archs.__all__
# LOSS_NAMES = losses.__all__
# LOSS_NAMES.append('BCEWithLogitsLoss')

import USwin_Haar2_CAETFM2
# ====== 对比试验========================================
# from Compare_models.fcn.fcn import FCN, fcn_resnet50
# from Compare_models.deeplab.deeplabv3_plus import DeepLab
# from Compare_models.segformer.segformer import SegFormer
# # ------------
# from Compare_models.SegNet.SegNet import SegNet
# from Compare_models.SegUKAN.SegUKAN import SegUKAN
# from Compare_models.UnetPlusPlus.UnetPlusPlus import UnetPlusPlus
# from Compare_models.Att_Unet.Att_Unet import AttU_Net
# from Compare_models.TransUNet.networks.vit_seg_modeling import VisionTransformer # TransUNet
# from Compare_models.TransUNet.networks.vit_seg_configs import get_b16_config # TransUNet根据需要导入不同的配置函数

import USwin_Haar2_CAETFM2

LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list

def parse_args():
    parser = argparse.ArgumentParser()
    # 1 General settings 通用设置 模型名 训练轮数 批量大小 划分数据的随机种子
    # --name：模型名称，默认为 None，如果 用户没有在命令行提供--name，则是None。help 是一个字符串，用于在帮助文档中显示参数的描述
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    # --dataseed：数据划分的随机种子，默认为 2981。
    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='UNetModel')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='DeepLab')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='FCN')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='SegFormer')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='fcn_resnet50')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='SegNet') #SegNet
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='SegUKAN') #SegUKAN
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='UnetPlusPlus')  # UnetPlusPlus
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='AttU_Net')  # AttU_Net
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='SwinTransformerSys') #Swin_Unet
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='VisionTransformer') #TransUNet 要用这个LovaszHingeLoss
    # parser.add_argument('--classifier', default='seg', help='Type of classifier to use')#TransUNet
    # 换模型时候修改
    parser.add_argument('--input_list', type=list_type, default=[256,320,512])

    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')
    # 4 dataset
    # ==============================================================================================
    # ===============================================================================================
    parser.add_argument('--dataset', default='CrackData_HL512_2528', help='dataset name')
    parser.add_argument('--data_dir', default=r'D:\Ahouli\OurUKANtest\input',
                        help='dataset dir')
    parser.add_argument('--output_dir', default=r'D:\Ahouli\OurUKANtest\models',
                        help='ouput dir')
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    # --kan_lr：KAN 的初始学习率，默认为 1e-2。  --kan_weight_decay：KAN 的权重衰减，默认为 1e-4。
    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    # 其他设置
    # --num_workers：数据加载的线程数，默认为 4。  --no_kan：是否不使用 KAN，默认为 False。
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--no_kan', action='store_true')
    # 新加 ===================断点设置
    parser.add_argument('--resume', action='store_true',
                        help='resume training from the latest checkpoint')

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):

    avg_meters = {
        'loss': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'recall': AverageMeter(),
        'precision': AverageMeter(),
        'f1': AverageMeter(),
        'accuracy': AverageMeter()
    }

    model.train()
    # 3 创建进度条  显示训练进度
    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        # 5 将数据移到 GPU：
        input = input.cuda()
        target = target.cuda()
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                if isinstance(output, dict):
                    output = output['out']
                loss += criterion(output, target)
            loss /= len(outputs)
            iou, dice, tp, tn, fp, fn, precision, recall, f1_score, accuracy = indicators(outputs[-1], target)
        else:
            output = model(input)
            if isinstance(output, dict):
                output = output['out']
            loss = criterion(output, target)
            iou, dice, tp, tn, fp, fn, precision, recall, f1_score, accuracy = indicators(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        avg_meters['recall'].update(recall, input.size(0))
        avg_meters['precision'].update(precision, input.size(0))
        avg_meters['f1'].update(f1_score, input.size(0))
        avg_meters['accuracy'].update(accuracy, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
            ('recall', avg_meters['recall'].avg),
            ('precision', avg_meters['precision'].avg),
            ('f1', avg_meters['f1'].avg),
            ('accuracy', avg_meters['accuracy'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    # 返回日志
    log = OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
        ('recall', avg_meters['recall'].avg),
        ('precision', avg_meters['precision'].avg),
        ('f1', avg_meters['f1'].avg),
        ('accuracy', avg_meters['accuracy'].avg)
    ])
    return log

def validate(config, val_loader, model, criterion):

    avg_meters = {
        'loss': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'recall': AverageMeter(),
        'precision': AverageMeter(),
        'f1': AverageMeter(),
        'accuracy': AverageMeter()
    }

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():

        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            if config['deep_supervision']:
                outputs = model(input)

                if isinstance(outputs, dict):
                    outputs = outputs['out']

                output = outputs[-1]
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, tp, tn, fp, fn, precision, recall, f1_score, accuracy = indicators(output, target)
            else:
                output = model(input)

                if isinstance(output, dict):
                    output = output['out']
                loss = criterion(output, target)
                iou, dice, tp, tn, fp, fn, precision, recall, f1_score, accuracy = indicators(output, target)


            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['recall'].update(recall, input.size(0))
            avg_meters['precision'].update(precision, input.size(0))
            avg_meters['f1'].update(f1_score, input.size(0))
            avg_meters['accuracy'].update(accuracy, input.size(0))

            #  新加 收集预测概率和真实标签
            preds = torch.sigmoid(output).cpu().numpy()  # 确保输出是概率
            targets = target.cpu().numpy().astype(int)  # 确保目标是整数

            all_preds.append(preds)
            all_targets.append(targets)

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('recall', avg_meters['recall'].avg),
                ('precision', avg_meters['precision'].avg),
                ('f1', avg_meters['f1'].avg),
                ('accuracy', avg_meters['accuracy'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)

        pbar.close()
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return (OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
        ('recall', avg_meters['recall'].avg),
        ('precision', avg_meters['precision'].avg),
        ('f1', avg_meters['f1'].avg),
        ('accuracy', avg_meters['accuracy'].avg)
    ]),
            all_preds, all_targets)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    # 禁用 cuDNN 的自动选择算法。如果设置为 True，cuDNN 会在第一次运行时选择最快的卷积算法，但这会导致每次运行时选择的算法不同，从而影响结果的可重复性。
    torch.backends.cudnn.benchmark = False
    # 设置 cuDNN 的确定性模式，确保每次运行时使用相同的卷积算法，从而提高结果的可重复性。
    torch.backends.cudnn.deterministic = True

# 添加
class HistogramEqualization(ImageOnlyTransform):

    def apply(self, img, **params):
        # OpenCV expects BGR format for equalizeHist
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Apply histogram equalization on each channel individually
        r, g, b = cv2.split(img_bgr)
        r_eq = cv2.equalizeHist(r)
        g_eq = cv2.equalizeHist(g)
        b_eq = cv2.equalizeHist(b)
        img_eq_bgr = cv2.merge((r_eq, g_eq, b_eq))
        # Convert back to RGB format
        img_eq_rgb = cv2.cvtColor(img_eq_bgr, cv2.COLOR_BGR2RGB)
        return img_eq_rgb

def main():
    # 设置随机种子
    seed_torch()
    # 解析命令行参数并转换为字典
    config = vars(parse_args())

    exp_name = config.get('name') # 获取实验名称
    # 获取当前时间
    now = datetime.datetime.now()
    # 将当前时间格式化为字符串，例如 '202¾-11-16-17-14'
    timestamp = now.strftime('%Y%m%d-%H%M')

    # 构建新的实验名称
    # exp_name = f"{exp_name}_{timestamp}"

    if config['resume'] and exp_name is not None:

        exp_name = exp_name
    else:

        if exp_name:
            exp_name = f"{exp_name}_{timestamp}"
        else:
            exp_name = f"{config['dataset']}_{config['arch']}_{timestamp}"

    output_dir = config.get('output_dir') # 获取输出目录

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)
    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))  # 打印配置信息
    print('-' * 20)


    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion) # 定义损失函数
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True # 启用 cuDNN 的自动选择算法

    model = USwin_Haar2_CAETFM2.__dict__[config['arch']](config['num_classes'],
                                                         config['input_channels'],
                                                         config['deep_supervision'],
                                                         embed_dims=config['input_list'],)
    # ============对比实验中的 实例化模型，并传递所需参数=================
    # model = DeepLab(num_classes=config['num_classes'], backbone="mobilenet", pretrained=False, downsample_factor=16)
    # model = fcn_resnet50(aux=True, num_classes=config['num_classes'], pretrain_backbone=False)
    # model = SegFormer(num_classes=config['num_classes'], phi='b0', pretrained=False)
    # # --------------------创建 SwinTransformerSys 实例时，设置 img_size=512------------------------------------------
    # model = SegNet(input_channels=config['input_channels'], output_channels=config['num_classes']) # 替换 NUM_CLASSES 为你的类别数
    # model = SegUKAN(config['num_classes'], config['input_channels'],config['deep_supervision'],
    #                                        embed_dims=config['input_list'], no_kan=config['no_kan'])
    # model = UnetPlusPlus(num_classes=config['num_classes'], deep_supervision=False)
    # model = AttU_Net(in_channel=config['input_channels'], num_classes=config['num_classes'],
    #                                     channel_list=[64, 128, 256, 512, 1024],checkpoint=False, convTranspose=True )
    # model = SwinTransformerSys(
    #     img_size=512, patch_size=4, num_classes=config['num_classes'], input_chans=config['input_channels'],
    #     deep_supervision=config['deep_supervision'], embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
    #     # 各层的注意力头数
    #     window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,  # 注意力 Dropout 率
    #     drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False,  # 是否添加绝对位置编码
    #     patch_norm=True,  # 是否在 Patch Embedding 后添加归一化
    #     use_checkpoint=False,  # 是否使用检查点
    #     final_upsample="expand_first"  # 最终上采样方式
    # )  # Swin_Unet
    # --------------------TransUNet------------------------------------------
    # zero_head=False：指定是否将分类头的权重初始化为零。通常在预训练模型时设置为 False。
    # vis=False：指定是否启用可视化功能。如果不需要可视化注意力权重等信息，可以设置为 False。
    # config_VisionTransformer = get_b16_config()
    # model = VisionTransformer(config_VisionTransformer, img_size=512, num_classes=1,
    #                           zero_head=False, vis=False)#vit_seg_modeling  TransUNet

    model = model.cuda()

    # 定义参数组
    param_groups = []

    kan_fc_params = []
    other_params = []

    for name, param in model.named_parameters():
        # print(name, "=>", param.shape) # 对特定层设置更高的学习率
        if 'layer' in name.lower() and 'fc' in name.lower(): # higher lr for kan layers # 对特定层设置更高的学习率
            # kan_fc_params.append(name)
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']})
        else:
            # other_params.append(name)
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})

    # st() 定义优化器
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'],
                              weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # 定义学习率调度器
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
    shutil.copy2('USwin_Haar2_CAETFM2.py', f'{output_dir}/{exp_name}/')


    dataset_name = config['dataset']
    
    img_ext = '.png'
    mask_ext ='.png'

    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    print(f"Number of images found: {len(img_ids)}")

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    train_transform = Compose([
        RandomRotate90(),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Resize(config['input_h'], config['input_w']),
        HistogramEqualization(),  # 使用自定义的直方图均衡化类
        # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),#默认值
        # Normalize(mean=(0.4674836, 0.47036508, 0.4726408), std=(0.14326131, 0.1428138, 0.1411014),
        #           max_pixel_value=255.0),
        Normalize(), #Normalize的使用：Normalize通常需要三个参数：均值（mean）、标准差（std）和图像的通道数。如果使用默认值，确保它们与您的数据集相匹配。
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        # transforms.Normalize(),
        HistogramEqualization(),  # 使用自定义的直方图均衡化类
        # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),#默认值
        # Normalize(mean=(0.4674836, 0.47036508, 0.4726408), std=(0.14326131, 0.1428138, 0.1411014),
        #           max_pixel_value=255.0),
        Normalize(),
    ])

    # 读取训练和验证数据
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'] ,config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    # 使用 DataLoader 来创建可迭代的数据加载器：
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config['batch_size'],
        shuffle=True,num_workers=config['num_workers'],drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=config['batch_size'],
        shuffle=False,num_workers=config['num_workers'],drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('recall', []),
        ('precision', []),
        ('f1', []),
        ('accuracy', [])
    ])

    best_iou = 0
    best_dice= 0
    trigger = 0

    # 检查是否存在检查点文件
    checkpoint_path = f'{output_dir}/{exp_name}/checkpoint.pth'
    if os.path.exists(checkpoint_path):
        print("=> loading checkpoint")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_iou = checkpoint['best_iou']
        best_dice = checkpoint['best_dice']
        trigger = checkpoint['trigger']
        log = checkpoint['log']  # 加载日志字典
        print(f"=> loaded checkpoint (epoch {start_epoch})")
    else:
        start_epoch = 0
        print("=> no checkpoint found")

    for epoch in range(start_epoch, config['epochs']):
        print('Epoch [%d/%d]' % (epoch + 1, config['epochs']))
        # train for one epoch为一个时代而训练
        train_log = train(config, train_loader, model, criterion, optimizer)

        val_log, all_preds, all_targets = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.5f - iou %.5f - val_loss %.5f - val_iou %.5f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['recall'].append(train_log['recall'])
        log['precision'].append(val_log['precision'])
        log['f1'].append(val_log['f1'])
        log['accuracy'].append(val_log['accuracy'])

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)
        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)

        try:

            fpr, tpr, _ = roc_curve(all_targets.flatten(), all_preds.flatten())
            roc_auc = auc(fpr, tpr)


            my_writer.add_scalar('val/auc', roc_auc, global_step=epoch)
        except ValueError as e:
            print(f"Error calculating ROC curve: {e}")
            continue



        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('IoU: %.5f' % best_iou)
            print('Dice: %.5f' % best_dice)

            # 绘制并保存 ROC 曲线
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(f'{output_dir}/{exp_name}/roc_curve_best.png')
            plt.close()
            # 重置 trigger
            trigger = 0
        else:
            trigger += 1

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

        checkpoint_path = f'{output_dir}/{exp_name}/checkpoint.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_iou': best_iou,
            'best_dice': best_dice,
            'trigger': trigger,
            'log': log  # 保存日志字典
        }, checkpoint_path)

if __name__ == '__main__':
    main()
