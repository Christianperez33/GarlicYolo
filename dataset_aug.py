import numpy as np
import os, png
import tensorflow as tf
import copy
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def normalize(image):
    return image / 255.

def rgb2gray(rgb):
    res = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return res

def rgb2gray_2(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def resize_normalize(image, size=(64, 64)): 
    resized = ktf.image.resize_images(image, size)
    resized = Reshape(size + (1,), input_shape=size)(resized)
    return resized

def random_crop(img, random_crop_size):
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx)]



dirr_img = "./dataset/images/"
images = np.array([cv2.imread(dirr_img+img,0) for img in sorted(os.listdir(dirr_img))])
output = images[0].shape
print("Dataset loaded...")

dirr_background = "./dataset/background/"
background = np.array([cv2.imread(dirr_background+img,0) for img in sorted(os.listdir(dirr_background))])
print("Backgroung loaded...")


dirr = "./dataset/Labelbox/"
labeledBoxs = sorted(os.listdir(dirr))
last_name = labeledBoxs[-1][:-4]
dirr_out = "./dataset/output/"
for i in range(len(labeledBoxs)-1):
    label = labeledBoxs[i]
    imagen = images[i]
    ##getting background image
    bgimg = random_crop(background[random.randint(0, len(background)-1)],output)

    filer = open(dirr+label)
    lines = filer.readlines()
    filer.close()
    number = int(last_name[-7:])+1
    last_name = 'image-'+str('0'*(7-len(str(number))))+str(number)
    pngImage = dirr_out+last_name+".png"
    txtImage = dirr_out+last_name+".txt"    
    with open(txtImage, "w") as f1:
        for j in lines:
            f1.write(j)
            coor = [float(i) for i in j.split(" ")]
            xcentro = int(round(bgimg.shape[1]*coor[1]))
            ycentro = int(round(bgimg.shape[0]*coor[2]))
            w = int(round(bgimg.shape[1]*coor[3]))//2
            h = int(round(bgimg.shape[0]*coor[4]))//2
            xminv = xcentro-w
            yminv = ycentro-h
            xmaxv = xcentro+w
            ymaxv = ycentro+h
            bgimg[yminv:ymaxv,xminv:xmaxv] = imagen[yminv:ymaxv,xminv:xmaxv]

    png.from_array(bgimg, 'L').save(pngImage)
    print("{} - {:2f}%".format((pngImage),((i/len(labeledBoxs))*100)))