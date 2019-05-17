# -*- coding: utf-8 -*-
# Author: ouc
# Email: 1412hyde@gmail.com
# Time: 2019/5/16 19:21
# File name: test.py
# IDE: VisDrones2019-VD

import cv2
import glob
import os
from core.detectors import CornerNet_Saccade
from core.vis_utils import draw_bboxes
import numpy as np


val_pic_dir = '/home/ws/datasets/Vis/VisDrone2018-VID-val/sequences'
val_detect_dir = '/home/ws/datasets/Vis/VisDrone2018-VID-val/results'

VisCategory_dic = {
    "ignored-regions":0,
    "pedestrian":1,
    "people":2,
    "bicycle":3,
    "car":4,
    "van":5,
    "truck":6,
    'tricycle':7,
    'awning-tricycle':8,
    "bus":9,
    "motor":10,
    "others":11
}

def get_all_files(dir):
    return glob.glob(os.path.join(dir,'*'))

def detect_frame(pic_dir,detector):
    # 1,-1,95,503,42,49,0.9823,4,-1,-1
    # frame,-1,x,y,w,h,score,cate,-1,-1

    pic_frame = pic_dir[-11:-4]
    write_list = []
    image = cv2.imread(pic_dir)
    bboxes = detector(image)
    for k in bboxes.keys():
        for bbox in bboxes[k]:
            write_list.append(str(int(pic_frame))+',-1,'+
                              str(int(bbox[0]+0.5))+','+
                              str(int(bbox[1]+0.5))+','+
                              str(int(bbox[2]-bbox[0]+0.5))+','+
                              str(int(bbox[3]-bbox[1]+0.5))+','+
                              str(round(float(bbox[4])*100)/100.0)+','+
                              str(VisCategory_dic.get(k))+',-1,-1\n')
    return write_list

def write_txt(dir,list):
    f = open(dir, 'w')
    for l in list:
        f.writelines(l)
    f.close()

def del_n(dir):
    f = open(dir, "r+")
    f.seek(-1, os.SEEK_END)
    if f.next() == "\n":
        f.seek(-1, os.SEEK_END)
        f.truncate()
    f.close()

if __name__ == '__main__':
    val_dir_list = get_all_files(val_pic_dir)
    # print(val_dir_list[0][-18:])
    # print(val_dir_list[0])

    detector = CornerNet_Saccade()
    for item in val_dir_list:
        write_det_list = []
        print("val: ", item)
        pic_list = get_all_files(item)
        print(len(pic_list))
        for pic in pic_list:
            write_det_list.append(detect_frame(pic,detector))
        write_txt(os.path.join(val_detect_dir, item[-18:] + '.txt'),write_det_list)