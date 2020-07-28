import os,sys
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)

pj = os.path.join

import numpy as np
import torch
from torch.autograd import Variable
from src.get_nets import PNet, RNet, ONet
from src.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from src.first_stage import run_first_stage
import cv2
from src import detect_faces, show_bboxes
import argparse
from PIL import Image

is_cuda = torch.cuda.is_available()

class MTCNNInference(object):

    def __init__(self, weights_path, expand_ratio=0.2):
        super(MTCNNInference, self).__init__()
        self.expand_ratio = expand_ratio
        self.pnet, self.rnet, self.onet = self._load_model(weights_path)

    def _load_model(self,weights_path):
        # LOAD MODELS
        w_pnet_path = pj(weights_path,'pnet.npy')
        w_rnet_path = pj(weights_path,'rnet.npy')
        w_onet_path = pj(weights_path,'onet.npy')
        pnet = PNet(w_pnet_path)
        rnet = RNet(w_rnet_path)
        onet = ONet(w_onet_path)

        pnet.eval()
        rnet.eval()
        onet.eval()

        if is_cuda:
            pnet = pnet.cuda()
            rnet = rnet.cuda()
            onet = onet.cuda()

        return pnet, rnet, onet

    def _select_max_face(self, bboxes,landmarks, img_h, img_w):
        bboxes = np.asarray(bboxes)[:,0:4]
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        max_area_idx = np.argmax(areas)
        max_area_bbox = bboxes[max_area_idx]
        max_area_landmarks = landmarks[max_area_idx]
        x0, y0, x1, y1 = max_area_bbox
        w = x1 - x0
        h = y1 - y0
        xc = int((x0 + x1) / 2)
        yc = int((y0 + y1) / 2)
        size = max(w, h) + int(min(w, h) * self.expand_ratio)
        if xc + int(size/2) > img_w:
            size = (img_w - xc) * 2
        if xc - int(size/2) < 0:
            size = xc * 2
        if yc + int(size/2) > img_h:
            size = (img_h - yc) * 2
        if yc - int(size/2) < 0:
            size = yc * 2
        x0, y0 = xc - int(size / 2), yc - int(size / 2)
        x1, y1 = xc + int(size / 2), yc + int(size / 2)
        return (x0, y0, x1, y1),max_area_landmarks

    def _select_max_face_with_prob(self, bboxes2,landmarks, img_h, img_w):
        bboxes = np.asarray(bboxes2)[:,0:4]
        probs = np.asarray(bboxes2)[:,4]
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        max_area_idx = np.argmax(areas)
        max_area_bbox = bboxes[max_area_idx]
        max_area_landmarks = landmarks[max_area_idx]
        max_area_prob = probs[max_area_idx]

        x0, y0, x1, y1 = max_area_bbox
        w = x1 - x0
        h = y1 - y0
        xc = int((x0 + x1) / 2)
        yc = int((y0 + y1) / 2)
        size = max(w, h) + int(min(w, h) * self.expand_ratio)
        if xc + int(size/2) > img_w:
            size = (img_w - xc) * 2
        if xc - int(size/2) < 0:
            size = xc * 2
        if yc + int(size/2) > img_h:
            size = (img_h - yc) * 2
        if yc - int(size/2) < 0:
            size = yc * 2
        x0, y0 = xc - int(size / 2), yc - int(size / 2)
        x1, y1 = xc + int(size / 2), yc + int(size / 2)
        return (x0, y0, x1, y1),max_area_landmarks,max_area_prob

    def is_face_inside(self,img_cv2,
                 min_face_size=20.0,
                 thresholds=[0.6, 0.6, 0.6],
                 nms_thresholds=[0.7, 0.7, 0.7]):
        if img_cv2 is None:
            print('input is null')
            return -2

        if len(img_cv2.shape) < 3:
            img_cv2 = cv2.cvtColor(img_cv2,cv2.COLOR_GRAY2BGR)

        bbox,landmarks = self.run(img_cv2,min_face_size,thresholds,nms_thresholds)

        bbox = list(bbox)

        if len(bbox) > 0:
            return 1
        else:
            return 0


    def run(self,
                 img_cv2,
                 min_face_size=20.0,
                 thresholds=[0.6, 0.6, 0.6],
                 nms_thresholds=[0.7, 0.7, 0.7],
                 ):

        h, w = img_cv2.shape[0:2]

        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        # img_pil = Image.fromarray(img_cv2)
        # BUILD AN IMAGE PYRAMID
        width, height = img_pil.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        with torch.no_grad():
            # run P-Net on different scales
            for s in scales:
                boxes = run_first_stage(img_pil, self.pnet, scale=s, threshold=thresholds[0])
                bounding_boxes.append(boxes)

            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            if len(bounding_boxes) == 0:
                return [],[]

            bounding_boxes = np.vstack(bounding_boxes)

            keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
            bounding_boxes = bounding_boxes[keep]

            # use offsets predicted by pnet to transform bounding boxes
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            # shape [n_boxes, 5]

            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 2

            img_boxes = get_image_boxes(bounding_boxes, img_pil, size=24)
            img_boxes = torch.FloatTensor(img_boxes)

            output = self.rnet(img_boxes)
            output = [t.cpu() for t in output]
            offsets = output[0].data.numpy()  # shape [n_boxes, 4]
            probs = output[1].data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]

            keep = nms(bounding_boxes, nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 3

            img_boxes = get_image_boxes(bounding_boxes, img_pil, size=48)
            if len(img_boxes) == 0:
                return [], []
            img_boxes = torch.FloatTensor(img_boxes)
            output = self.onet(img_boxes)
            output = [t.cpu() for t in output]
            landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].data.numpy()  # shape [n_boxes, 4]
            probs = output[2].data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]

#         return bounding_boxes, landmarks
#         print(bounding_boxes)
        if len(bounding_boxes) == 0:
            return [],[]
        selected_bbox,selected_landmark = self._select_max_face(bounding_boxes,landmarks, h, w)
        return selected_bbox,selected_landmark

    def __call__(self,
            img_cv2,
            min_face_size=20.0,
            thresholds=[0.6, 0.6, 0.6],
            nms_thresholds=[0.7, 0.7, 0.7],
            ):
        return self.run(img_cv2,min_face_size,thresholds,nms_thresholds)




    def run_onet(self,img_cv2,bounding_boxes,
                 thresholds=0.6,
                 nms_thresholds=0.7):

        if isinstance(bounding_boxes,list):
            return [],[],[]

        h, w = img_cv2.shape[0:2]
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        # BUILD AN IMAGE PYRAMID
        width, height = img_pil.size

        with torch.no_grad():
            img_boxes = get_image_boxes(bounding_boxes, img_pil, size=48)
            if len(img_boxes) == 0:
                return [], []
            img_boxes = torch.FloatTensor(img_boxes)
            output = self.onet(img_boxes)
            output = [t.cpu() for t in output]
            landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].data.numpy()  # shape [n_boxes, 4]
            probs = output[2].data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds)[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds, mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]

            #         return bounding_boxes, landmarks
        print(bounding_boxes)
        if len(bounding_boxes) == 0:
            return [], [],[]
        selected_bbox, selected_landmark,selected_prob = self._select_max_face_with_prob(bounding_boxes, landmarks, h, w)
        return selected_bbox, selected_landmark,selected_prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',default='/Users/marschen/Ucloud/Data/error_analysis/emb_sys/image_cache/')
    parser.add_argument('--o',default='/Users/marschen/Ucloud/Data/error_analysis/emb_sys/image_cache_result/image_cache_fail/')
    args = parser.parse_args()

    weights_path = 'src/weights/'
    weights_path = os.path.abspath(weights_path)
    mtcnn_infer = MTCNNInference(weights_path= weights_path)

    datapath = args.i
    savepath = args.o

    img_lst = [ c for c in os.listdir(datapath) if not c.startswith('.')]
    total_img = len(img_lst)

    for img_name in img_lst:
        img_fpath = pj(datapath, img_name)
        img_cv2 = cv2.imread(img_fpath)

        ret = mtcnn_infer.is_face_inside(img_cv2,
                 min_face_size=50.0,
                 thresholds=[0.6, 0.6, 0.6],
                 nms_thresholds=[0.7, 0.7, 0.7])
        print(img_name,ret)

        if ret <= 0:
            cv2.imwrite(pj(savepath,img_name),img_cv2)

if __name__=='__main__':

    main()


