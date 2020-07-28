import os,sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

pj = os.path.join

import cv2,torch
import numpy as np
import argparse

from face_validation import face_validator

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',default='/Users/marschen/Ucloud/Data/error_analysis/badge_ids_sample/')
    parser.add_argument('--o',default='/Users/marschen/Ucloud/Data/error_analysis/emb_sys/image_cache_result/image_cache_fail/')
    args = parser.parse_args()

    print('--> loading mtcnn weights')
    weights_path = 'src/weights/'
    weights_path = os.path.abspath(weights_path)

    face_val = face_validator(weights_path=weights_path,iqa_threshold=100)

    datapath = args.i
    savepath = args.o

    img_lst = [ c for c in os.listdir(datapath) if not c.startswith('.')]
    total_img = len(img_lst)

    for img_name in img_lst:
        img_fpath = pj(datapath, img_name)
        img_cv2 = cv2.imread(img_fpath)

        ret = face_val(img_cv2)
        if ret['val'] <= 0:
            print(ret['info'])
            cv2.imwrite(pj(savepath,img_name),img_cv2)