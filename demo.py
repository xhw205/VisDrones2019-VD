#!/usr/bin/env python
import cv2
from core.detectors import CornerNet_Saccade
from core.vis_utils import draw_bboxes
import numpy as np
detector = CornerNet_Saccade()
image    = cv2.imread("classical.jpg")
bboxes = detector(image)
newbboxes = {}
for keys in bboxes.keys():
    area_all = 0.0
    num = 0
    areas = []
    scales = []
    for bbx in bboxes[keys]:
        num=num+1
        area = (bbx[2]-bbx[0])*(bbx[3]-bbx[1])
        scales.append(float( (bbx[2]-bbx[0])/(bbx[3]-bbx[1]) ))
        areas.append(area)
    if num == 0:
        newbboxes[keys] =  np.zeros((0,5), dtype=np.float32)
    else:
        areas = np.array(areas, dtype=np.float32)
        areas = sorted(areas)
        if len(areas)>=5:
            areas = np.delete(areas, 0, axis=0)
            areas = np.delete(areas, 0, axis=0)
            areas = np.delete(areas, len(areas) - 1, axis=0)
            areas = np.delete(areas, len(areas) - 1, axis=0)
            area_avg = areas.sum() / len(areas)
        else:
            area_avg = np.array(areas).sum()/len(areas)
        scales = np.array(scales, dtype=np.float32)
        scale_avg = scales.sum() / len(scales)
        dets = []
        for bbx in bboxes[keys]:
            area = (bbx[2] - bbx[0]) * (bbx[3] - bbx[1])
            scale = (bbx[2]-bbx[0])/(bbx[3]-bbx[1])
            if area <= 3.0*area_avg and scale<=1.5*scale_avg:
                dets.append(bbx)

        dets = np.vstack(dets)
        newbboxes[keys] = np.array(dets, dtype=np.float32)
image  = draw_bboxes(image, newbboxes)
cv2.imwrite("demo_out.jpg", image)
