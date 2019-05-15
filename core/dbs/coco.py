import os
import json
import numpy as np
import pandas as pd
from .detection import DETECTION
from ..paths import get_file_path
# COCO bounding boxes are 0-indexed
class COCO(DETECTION):
    def __init__(self, db_config, split=None, sys_config=None):
        assert split is None or sys_config is not None
        super(COCO, self).__init__(db_config)
        self._mean    = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std     = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self._coco_cls_ids = [
            1,2,3,4,5,6,7,8,9,10
        ]
        self._coco_cls_names = [
            'pedestrian', 'people', 'bicycle', 'car',
            'van', 'truck', 'tricycle', 'awning-tricycle',
            'bus','motor',
        ]
        self._cls2coco  = {ind + 1: coco_id for ind, coco_id in enumerate(self._coco_cls_ids)}
        self._coco2cls  = {coco_id: cls_id for cls_id, coco_id in self._cls2coco.items()}
        self._coco2name = {cls_id: cls_name for cls_id, cls_name in zip(self._coco_cls_ids, self._coco_cls_names)}
        self._name2coco = {cls_name: cls_id for cls_name, cls_id in self._coco2name.items()}
        if split is not None:
            coco_dir = os.path.join(sys_config.data_dir, "coco")
            self._split     = {
                "trainval": "train",
                "minival":  "val",
                "testdev":  "testdev2017"
            }[split]
            # self._data_dir  = os.path.join(coco_dir, "images", self._split)
            self._data_dir = './VisDrone2018-VID-train/sequences/'
            # self._anno_file = '/home/ws/datasets/Vis/train.json'
            # self._img_path = '/home/ws/datasets/Vis/VisDrone2018-VID-train/sequences/'
            # self._ann_path = '/home/ws/datasets/Vis/train/annotations/'
            # self.image_ids = os.listdir(self._img_path)
            self._detections, self._eval_ids = self._load_coco_annos()
            self._image_ids = list(self._detections.keys())
            self._db_inds   = np.arange(len(self._image_ids))


    def _load_coco_annos(self):
        print('Strating to create index.....')
        import time
        st = time.time()
        class_ids = [
            1,2,3,4,5,6,7,8,9,10
        ]
        eval_ids = {}
        detections = {}
        with open(os.path.join(os.path.abspath('./'),'alltrain.txt'), 'r') as fh:
            for line in fh.readlines():
                words = line.strip('\n').strip().split(',')
                img_name = words[0]
                dets = []
                for word in words[1:-1]:
                    word = word.split(' ')
                    x, y, x2, y2, label = int(word[0]), int(word[1]), int(word[2]), int(word[3]), int(word[4])
                    bboxs = np.array([x, y, x2, y2, label])
                    dets.append(bboxs)
                dets = np.vstack(dets)
                # if img_name in detections.keys():
                #     detections[img_name] = np.zeros((0,5), dtype=np.float32)
                # else:
                detections[img_name] = np.array(dets, dtype=np.float32)
            print('Uinsg %f' %(time.time() - st))
            print(detections)
            return detections, eval_ids

    def image_path(self, ind):
        if self._data_dir is None:
            raise ValueError("Data directory is not set")

        db_ind    = self._db_inds[ind]
        file_name = self._image_ids[db_ind]
        file_name = os.path.join(self._data_dir, file_name)
        return file_name

    def detections(self, ind):
        db_ind    = self._db_inds[ind]
        file_name = self._image_ids[db_ind]
        return self._detections[file_name].copy()

    def cls2name(self, cls):
        coco = self._cls2coco[cls]
        return self._coco2name[coco]


    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._cls2coco[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox  = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids):
        from pycocotools.cocoeval import COCOeval

        if self._split == "testdev":
            return None

        coco = self._coco

        eval_ids = [self._eval_ids[image_id] for image_id in image_ids]
        cat_ids  = [self._cls2coco[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]

