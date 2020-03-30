import os
import skimage.io
import skimage.transform
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_img(path):
    """
    输入图片所在的路径+名称
    从path位置读入单张的图片到img，
    将数据归一化，
    从图片中心点裁剪，成为正方形，
    重整图片尺寸（m, 宽, 高, channel）
    return：一张shape为(1, 224, 224, 3)的图片
    """
    #img = skimage.io.imread(path)
    img = cv2.imread(path)
    img.astype(np.float32)
    # img = img / 255.0
    resized_img = cv2.resize(img, (224, 224))
    out_img = preprocess_input(resized_img)
    # we crop image from center
    #short_edge = min(img.shape[:2])
    #yy = int((img.shape[0] - short_edge) / 2)
    #xx = int((img.shape[1] - short_edge) / 2)
    #crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    #resized_img = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
    return out_img
    
def load_data(prepath, num=None):
    """
    输入批量图片所在路径
   按照文件夹名称命名分类的name
    """
    classes = os.listdir(prepath)
    imgs = { cla: [] for cla in classes}
    y = []
    i = 0
    for k in imgs.keys():
        dir = prepath+ '/' + k
        for file in os.listdir(dir):
            if not file.lower().endswith('.jpg'):
                continue
            try:
                resized_img = load_img(os.path.join(dir, file))
            except OSError:
                continue
            imgs[k].append(resized_img)    # [1, height, width, depth] * n
            y.append(i)
            if len(imgs[k]) == num:        
                break
        i+=1
    return imgs, y