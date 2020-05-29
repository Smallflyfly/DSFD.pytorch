#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from models.factory import build_net
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='dsfd demo')
parser.add_argument('--network',
                    default='vgg', type=str,
                    choices=['vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--save_dir',
                    type=str, default='./mask_pic/',
                    help='Directory for detect result')
parser.add_argument('--model',
                    type=str,
                    default='weights/dsfd_face.pth', help='trained model')
parser.add_argument('--thresh',
                    default=0.6, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

global name_count
global name_ss


def detect(net, img_path, thresh):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1500 * 1000 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()
    t1 = time.time()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    # 图片类型
    img_type = img_path.split('.')[-1]
    # 原图
    origin_img = Image.open(img_path)

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            # score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
            # left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            xmin, ymin, xmax, ymax = pt[0], pt[1], pt[2], pt[3]
            xmin = max(0, xmin-5)
            ymin = max(0, ymin-5)
            xmax = min(width, xmax+5)
            ymax = min(height, ymax+5)
            crop_img = origin_img.crop((xmin, ymin, xmax, ymax))
            crop_img = crop_img.convert('RGB')
            global name_count
            name_count += 1
            filename = (name_ss + str(name_count))[-6:]
            print(args.save_dir + filename + img_type)
            crop_img.save(args.save_dir + filename + '.' + img_type)
            j += 1
            # cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 1)
            # conf = "{:.2f}".format(score)
            # text_size, baseline = cv2.getTextSize(
            #     conf, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            # p1 = (left_up[0], left_up[1] - text_size[1])
            # cv2.rectangle(img, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
            #               (p1[0] + text_size[0], p1[1] + text_size[1]),[255,0,0], 1)
            # cv2.putText(img, conf, (p1[0], p1[
            #                 1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, 8)

    # t2 = time.time()
    # print('detect:{} timer:{}'.format(img_path, t2 - t1))

            # cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)


if __name__ == '__main__':
    net = build_net('test', cfg.NUM_CLASSES, args.network)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    name_count = 0
    name_ss = '000000'

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    img_path = './masks'
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('jpg') or x.endswith('jpeg')]
    for path in img_list:
        detect(net, path, args.thresh)
