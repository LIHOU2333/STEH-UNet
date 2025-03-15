#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
from glob import glob
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset

import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.dataset import Dataset
from albumentations import Resize

from PIL import Image

from metrics2 import indicators
from utils.utils import AverageMeter

import USwin_Haar2_CAETFM2
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default=r'D:\Ahouli\OurUKANtest\models',
                        help='ouput dir')
    # 解析命令行参数
    args = parser.parse_args()

    return args
#  设置随机种子
def seed_torch(seed=1029):
    random.seed(seed)  #random.seed(seed)：设置 Python 内置的随机数生成器的种子，确保生成的随机数序列是可复现的。
    os.environ['PYTHONHASHSEED'] = str(seed) #设置 Python 的哈希种子。这会影响字典和集合等数据结构的内部哈希函数，确保它们的行为是确定的。
    np.random.seed(seed) # 设置 NumPy 的随机数生成器的种子，确保生成的随机数序列是可复现的。
    torch.manual_seed(seed) # 设置 PyTorch 的 CPU 随机数生成器的种子，确保生成的随机数序列是可复现的。
    torch.cuda.manual_seed(seed) # 设置 PyTorch 的单个 GPU 随机数生成器的种子，确保生成的随机数序列是可复现的。
    torch.cuda.manual_seed_all(seed) # 设置 PyTorch 的所有 GPU 随机数生成器的种子，确保生成的随机数序列是可复现的。这对于多 GPU 训练特别重要。
    torch.backends.cudnn.benchmark = False # 禁用 cuDNN 的自动优化。cuDNN 会根据输入数据的大小选择最优的卷积算法，但这种选择是不可复现的。禁用后，cuDNN 将使用固定的算法，确保结果的可复现性。
    torch.backends.cudnn.deterministic = True # 启用 cuDNN 的确定性模式。这确保了卷积操作的结果是确定的，即使代价是性能的下降。

#  主函数
def main():
    seed_torch()  # ：调用 seed_torch 函数，设置各种随机数生成器的种子，确保结果的可复现性。
    args = parse_args()

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)
    # 启用 cuDNN 的自动优化，提高卷积操作的性能。注意，这与 seed_torch 中的 cudnn.benchmark = False 相矛盾，这里可能是为了性能考虑而重新启用。
    cudnn.benchmark = True

    model = USwin_Haar2_CAETFM2.__dict__[config['arch']](config['num_classes'],
                                                   config['deep_supervision'])

    # model = UNet(n_channels=3, n_classes=1)
    #=================================================== 对比模型=========================================
    # model = DeepLab(num_classes=config['num_classes'], backbone="mobilenet", pretrained=False, downsample_factor=16)
    # model = fcn_resnet50(aux=True, num_classes=config['num_classes'], pretrain_backbone=False)
    # model = SegFormer(num_classes=config['num_classes'], phi='b0', pretrained=False)
    # model = SegNet(input_channels=config['input_channels'], output_channels=config['num_classes'])  # 替换 NUM_CLASSES 为你的类别数
    # model = SegUKAN(config['num_classes'], config['input_channels'],config['deep_supervision'],
    #                                        embed_dims=config['input_list'], no_kan=config['no_kan'])
    # model = UnetPlusPlus(num_classes=config['num_classes'], deep_supervision=False)
    # model = AttU_Net(
    #     in_channel=config['input_channels'],  # 输入图像的通道数（例如RGB图像为3）
    #     num_classes=config['num_classes'],  # 输出类别数量（对于二分类问题通常为1）
    #     channel_list=[64, 128, 256, 512, 1024],  # 各层的通道数列表
    #     checkpoint=False,  # 是否加载检查点，这里假设不加载
    #     convTranspose=True  # 使用转置卷积进行上采样
    # )

    # config_VisionTransformer = get_b16_config()
    # model = VisionTransformer(config_VisionTransformer, img_size=512, num_classes=1, zero_head=False,
    #                           vis=False)  # vit_seg_modeling  TransUNet

    model = model.cuda()
    #  7 确定数据集和扩展名 根据配置文件中的 dataset 参数，确定数据集的名称和对应的图像及标签文件扩展名
    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'Comprehensive_Crack_Data':
        mask_ext = '.png'
    elif dataset_name == 'CrackData_HL512_2528':
        mask_ext = '.png'
    elif dataset_name == 'CrackData_HL512512':
        mask_ext = '.png'

    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids] # 用于从文件路径中提取文件名（不包括扩展名）

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth')
    try:
        model.load_state_dict(ckpt)
    except:
        print("Pretrained model keys:", ckpt.keys())
        print("Current model keys:", model.state_dict().keys())

        pretrained_dict = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        current_dict = model.state_dict()
        diff_keys = set(current_dict.keys()) - set(pretrained_dict.keys())

        print("Difference in model keys:")
        for key in diff_keys:
            print(f"Key: {key}")

        model.load_state_dict(ckpt, strict=False)

    model.eval()


    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])


    val_dataset = Dataset(
        img_ids=val_img_ids,  # 使用 val_img_ids 作为验证集的图像 ID 列表。

        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),

        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],

        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()  # 初始化 IoU 指标追踪器
    dice_avg_meter = AverageMeter()  # 初始化 Dice 指标追踪器
    precision_avg_meter = AverageMeter()  # 初始化 Precision 指标追踪器
    recall_avg_meter = AverageMeter()  # 初始化 Recall 指标追踪器
    f1_avg_meter = AverageMeter()  # 初始化 F1-Score 指标追踪器
    accuracy_avg_meter = AverageMeter()  # 初始化 Accuracy 指标追踪器


    with torch.no_grad():

        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()

            output = model(input)

            iou, dice, tp, tn, fp, fn, precision, recall, f1_score, accuracy = indicators(output, target)

            # 更新指标
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            precision_avg_meter.update(precision, input.size(0))
            recall_avg_meter.update(recall, input.size(0))
            f1_avg_meter.update(f1_score, input.size(0))
            accuracy_avg_meter.update(accuracy, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
            for pred, img_id in zip(output, meta['img_id']):
                pred_np = pred[0].astype(np.uint8)
                pred_np = pred_np * 255
                img = Image.fromarray(pred_np, 'L')
                img.save(os.path.join(args.output_dir, config['name'], 'out_val/{}.jpg'.format(img_id)))



    print(config['name'])
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('Precision: %.4f' % precision_avg_meter.avg)
    print('Recall: %.4f' % recall_avg_meter.avg)
    print('F1-score: %.4f' % f1_avg_meter.avg)
    print('Accuracy: %.4f' % accuracy_avg_meter.avg)

    results_path = os.path.join(args.output_dir, config['name'], 'validation_results.txt')


    with open(results_path, 'w') as f:
        f.write(f"Experiment Name: {config['name']}\n")
        f.write(f"IoU: {iou_avg_meter.avg:.4f}\n")
        f.write(f"Dice: {dice_avg_meter.avg:.4f}\n")
        f.write(f"Precision: {precision_avg_meter.avg:.4f}\n")
        f.write(f"Recall: {recall_avg_meter.avg:.4f}\n")
        f.write(f"F1-score: {f1_avg_meter.avg:.4f}\n")
        f.write(f"Accuracy: {accuracy_avg_meter.avg:.4f}\n")

#  入口点
if __name__ == '__main__':
    main()
