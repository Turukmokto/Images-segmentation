import json
import os.path as osp

import mmcv
import numpy as np
from mmdet.apis import inference_detector, show_result_pyplot
from pycocotools import coco
import matplotlib.pyplot as plt

from ann_creator import create_ann_file
from model import DetectorCreator

from utils import find_best_metrics
from utils import metrics_update
from utils import print_metrics

device = 'cuda'  # поменять на gpu


def main():
    config_file = './configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
    classes = ['road sign']
    work_dir = './res2'
    check_point = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

    creator = DetectorCreator(config_file, classes)
    create_ann_file('../sign_dataset/train/via_region_data.json','train_ann.json','../sign_dataset/train/')
    creator.add_train_path('train_ann.json', '../sign_dataset/train/')
    create_ann_file('../sign_dataset/val/via_region_data.json','val_ann.json','../sign_dataset/val/')
    creator.add_test_path('val_ann.json', '../sign_dataset/val/')
    creator.add_val_path('val_ann.json', '../sign_dataset/val/')

    creator.define_checkpoint_work_dir(check_point, work_dir)
    creator.define_learning_params(8, 1, 1, device)

    creator.train_model(work_dir)
    model = creator.get_model()

    for i in range(4):
        image = mmcv.imread(f'my_photos/{i}.jpg')
        model.cfg = creator.get_config()
        ans = inference_detector(model, image)
        show_result_pyplot(model, image, ans)

    infos = json.load(open('val_ann.json'))
    df = coco.COCO('val_ann.json')
    predicts, targets = [], [[] for _ in infos['images']]
    for info in infos['images']:
        img_path = osp.join('../sign_dataset/val/', info['file_name'])
        img = mmcv.imread(img_path)
        model.cfg = creator.get_config()
        result = inference_detector(model, img)
        predicts.append(result[1][0])
    for ann in infos['annotations']:
        targets[ann['image_id']].append(df.annToMask(ann))

    iou, precision, recall = [], [], []
    for i in range(len(predicts)):
        cur_max_iou, cur_precisions, cur_recalls = [], [], []
        for j in range(len(targets[i])):
            cur_max_iou, cur_precisions, cur_recalls = find_best_metrics(
                cur_max_iou, cur_precisions, cur_recalls,
                i, j,
                predicts, targets)

        metrics_update(cur_max_iou, cur_precisions, cur_recalls, iou, precision, recall)

    print_metrics(predicts, iou, precision, recall)


if __name__ == '__main__':
    main()
