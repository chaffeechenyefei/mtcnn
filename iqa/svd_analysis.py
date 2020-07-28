import cv2,math
import numpy as np
from sklearn.preprocessing import normalize


def im2col(mtx, block_size,stepsize=1):
    '''
    block_size = W(H)
    '''
    mtx_shape = mtx.shape
    sx = math.floor((mtx_shape[0] - block_size + 1)/stepsize)
    sy = math.floor((mtx_shape[1] - block_size + 1)/stepsize)
    # 如果设A为m×n的，对于[p q]的块划分，最后矩阵的行数为p×q，列数为(m−p+1)×(n−q+1)。
    result = np.empty((block_size,block_size, sx * sy))
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[:,:, i * sx + j] = mtx[j*stepsize:j*stepsize + block_size, i*stepsize:i*stepsize + block_size]
    return result

def col2im(mtx, image_size, block_size,stepsize=1):
    sx = math.floor((image_size[0] - block_size + 1) / stepsize)
    sy = math.floor((image_size[1] - block_size + 1) / stepsize)
    result = np.zeros(image_size)
    weight = np.zeros(image_size)+1e-3  # weight记录每个单元格的数字重复加了多少遍
    col = 0
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[j*stepsize:j*stepsize + block_size, i*stepsize:i*stepsize + block_size] += mtx[:,:, col]
            weight[j*stepsize:j*stepsize + block_size, i*stepsize:i*stepsize + block_size] += np.ones(block_size)
            col += 1
    return result / weight

def col2im_mtx_single_value(mtx, image_size, block_size, stepsize=1):
    sx = math.floor((image_size[0] - block_size + 1) / stepsize)
    sy = math.floor((image_size[1] - block_size + 1) / stepsize)

    nimage_size = [(sx - 1) * stepsize + block_size, (sy - 1) * stepsize + block_size]
    result = np.zeros(nimage_size)
    weight = np.zeros(nimage_size) + 1e-3  # weight记录每个单元格的数字重复加了多少遍
    col = 0
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[j * stepsize:j * stepsize + block_size, i * stepsize:i * stepsize + block_size] += mtx[col]
            weight[j * stepsize:j * stepsize + block_size, i * stepsize:i * stepsize + block_size] += np.ones(
                block_size)
            col += 1
    return result / weight

def compute_smap(img,blocksize,stepsize):
    patches = im2col(img,blocksize,stepsize)
    imgsize = img.shape
    n_patches = patches.shape[-1]
    mtx = []
    for n in range(n_patches):
        p = patches[:,:,n]
        s= np.linalg.svd(p,full_matrices=1,compute_uv=0)
        s = s[0]/s.sum()
        mtx.append(s)
    mtx = np.array(mtx)
    smap = col2im_mtx_single_value(mtx,imgsize,blocksize,stepsize)
    return smap

def get_texture_score(img,blocksize,stepsize):
    smap = compute_smap(img,blocksize,stepsize)
    return smap.mean()

def face_format(img_cv2, format_size=112):
    org_h, org_w = img_cv2.shape[0:2]
    rescale_ratio = format_size / max(org_h, org_w)
    h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
    img_rescaled = cv2.resize(img_cv2, (w, h))
    paste_pos = [int((format_size - w) / 2), int((format_size - h) / 2)]
    img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
    img_format[paste_pos[1]:paste_pos[1] + h, paste_pos[0]:paste_pos[0] + w] = img_rescaled
    return img_format


def crop_margin(img_cv, margin=0.2):
    h, w = img_cv.shape[0:2]
    margin = math.floor(margin * min(h, w))

    crop_img_cv = img_cv[margin:h - margin, margin:w - margin].copy()
    return crop_img_cv


def crop_margin_col(img_cv, margin=0.2):
    h, w = img_cv.shape[0:2]
    margin = math.floor(margin * min(h, w))

    crop_img_cv = img_cv[:, margin:w - margin].copy()
    return crop_img_cv

def get_blur_score(img_cv):
    return cv2.Laplacian(img_cv,ddepth=cv2.CV_64FC1).var()