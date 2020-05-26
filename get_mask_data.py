# -*- coding: utf-8 -*-
# @Time    : 2020/5/26 20:04
# @Author  : Fangpf
# @FileName: get_mask_data.py

import os
import shutil

name_count = 0
name_str = '000000'

path = './'
save_image_path = './images/'
list_fold = os.listdir(path)
for fold in list_fold:
    new_path = os.path.join(path+fold)
    imgs = os.listdir(new_path)
    for img in imgs:
        img_type = img.split('.')[-1]
        img_name = (str(name_count) + name_str)[-6:]
        shutil.copy(os.path.join(path, new_path, img), save_image_path + img_name + '.' + img_type)
        name_count += 1

print('Done')
