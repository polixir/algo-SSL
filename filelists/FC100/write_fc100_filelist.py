# -*- encoding: utf-8 -*-
'''
File    :   write_fc100_filelist.py
Time    :   2020/03/09 12:45:34
Author  :   Chao Wang 
Version :   1.0
Contact :   374494067@qq.com
@Desc    :   None
'''

import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re

cwd = os.getcwd() 
data_path = join(cwd,'./FC100')
savedir = './'
dataset_list = ['base', 'val', 'novel']



cl = -1
folderlist = []

datasetmap = {'base':'train','val':'val','novel':'test'};
filelists = {'base':{},'val':{},'novel':{} }
filelists_flat = {'base':[],'val':[],'novel':[] }
labellists_flat = {'base':[],'val':[],'novel':[] }

for dataset in dataset_list:
    THE_PATH = join(data_path, datasetmap[dataset])
    label_list = os.listdir(THE_PATH)
    for label in label_list:
        if label  != '.DS_Store':
            folderlist.append(label)
            this_folder = join(THE_PATH, label)
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                if image_path  != '.DS_Store':
                    fname = join(this_folder, image_path)
                    if not label in filelists[dataset]:
                        filelists[dataset][label] = [fname]
                    else:
                        filelists[dataset][label].append(fname)


    for key, filelist in filelists[dataset].items():
        cl += 1
        random.shuffle(filelist)
        filelists_flat[dataset] += filelist
        labellists_flat[dataset] += np.repeat(cl, len(filelist)).tolist() 


for dataset in dataset_list:
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folderlist])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in filelists_flat[dataset]])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in labellists_flat[dataset]])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)