import os

pj = os.path.join

import cv2
import numpy as np


from mtcnn_inference import MTCNNInference
from iqa.svd_analysis import get_texture_score,face_format,crop_margin_col

class face_validator(object):
    def __init__(self,weights_path,iqa_threshold=80):
        self._load_model(weights_path=weights_path)
        self.iqa_threshold = iqa_threshold


    def _load_model(self,weights_path):
        self.mtcnn_infer = MTCNNInference(weights_path=weights_path)


    def __call__(self, img_cv2):
        if len(img_cv2.shape) < 3:
            img_cv2 = cv2.cvtColor(img_cv2,cv2.COLOR_GRAY2BGR)

        bbox, landmark = self.mtcnn_infer(img_cv2,
                                     min_face_size=50.0,
                                     thresholds=[0.6, 0.6, 0.6],
                                     nms_thresholds=[0.7, 0.7, 0.7])

        bbox = list(bbox)
        if len(bbox) == 0:
            return {
                'val':-1,
                'info':'no face detected'
            }

        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        w = int((bbox[2] - bbox[0]) / 2)
        h = int((bbox[3] - bbox[1]) / 2)

        img_face = cv2.getRectSubPix(img_cv2, (w, h), (cx, cy))
        img_face = face_format(img_face, 112)
        img_gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.astype(np.float32) / 255
        img_crop = crop_margin_col(img_gray, margin=0.2)
        score = get_texture_score(img_crop, blocksize=30, stepsize=10)

        if score > self.iqa_threshold:
            return {
                'val':-2,
                'info':'image quality no good enough'
            }

        return {
                'val':1,
                'info':'accept'
            }



