import os

pj = os.path.join

import cv2
import numpy as np


from mtcnn_inference import MTCNNInference
from iqa.svd_analysis import get_blur_score,face_format,crop_margin

class face_validator(object):
    def __init__(self,weights_path,iqa_threshold=120):
        self._load_model(weights_path=weights_path)
        self.iqa_threshold = iqa_threshold


    def _load_model(self,weights_path):
        self.mtcnn_infer = MTCNNInference(weights_path=weights_path)


    def __call__(self, img_cv2):
        if len(img_cv2.shape) < 3:
            img_cv2 = cv2.cvtColor(img_cv2,cv2.COLOR_GRAY2BGR)

        hh,ww = img_cv2.shape[:2]
        det_sz = 1024
        scale = det_sz/min([hh,ww])

        scale = max([scale,1.0])
        img_cv2_scaled = cv2.resize(img_cv2,dsize=None,fx=scale,fy=scale)


        bbox, landmark = self.mtcnn_infer(img_cv2_scaled,
                                     min_face_size=50.0,
                                     thresholds=[0.5, 0.6, 0.6],
                                     nms_thresholds=[0.7, 0.7, 0.7])

        bbox = list(bbox)
        if len(bbox) == 0:
            return {
                'val':-1,
                'info':'no face detected'
            }

        cx = int((bbox[0] + bbox[2]) / 2 / scale)
        cy = int((bbox[1] + bbox[3]) / 2 / scale)
        w = int((bbox[2] - bbox[0])/scale)
        h = int((bbox[3] - bbox[1])/scale)

        img_face = cv2.getRectSubPix(img_cv2, (w, h), (cx, cy))
        img_face = face_format(img_face, 112)
        # cv2.imwrite('test.jpg',img_face)
        img_gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
        # img_gray = img_gray.astype(np.float32) / 255
        img_crop = crop_margin(img_gray, margin=0.2)
        score = get_blur_score(img_crop)
        # score = get_texture_score(img_crop, blocksize=30, stepsize=10)

        if score < self.iqa_threshold:
            print(score)
            return {
                'val':-2,
                'info':'image quality no good enough'
            }

        return {
                'val':1,
                'info':'accept'
            }



