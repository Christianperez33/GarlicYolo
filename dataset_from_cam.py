from __future__ import print_function
from WebcamVideoStreamFPS import webcamVideoStreamFPS
from imutils.video import FPS
from skimage.filters import threshold_otsu
from operator import itemgetter, attrgetter, methodcaller
import argparse
import collections
import itertools
import math
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os, sys

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=120,help="Number of frames to adquire (def 120FPS).")
ap.add_argument("-d", "--display", type=int, default=-1,help="Value of display (-1 debug, 1 operator view).")
ap.add_argument("-w", "--width", type=int, default=480,help="Size of captured window.")
ap.add_argument("-b", "--blockSize", type=int, default=2,help="Size of block for corner detector.")
ap.add_argument("-k", "--doublek", type=float, default=0.04,help="Value of constant k in Harris corner detector.")
ap.add_argument("-a", "--apertureSize", type=int, default=5,help="Apertire size of sobel algoritm in Harris corner detector.")
ap.add_argument("-s", "--listSize", type=int, default=5,help="Size of gralic angle list.")
args = vars(ap.parse_args())

doublek = args["doublek"]
blockSize = args["blockSize"]
apertureSize = args["apertureSize"]
angleList=collections.deque(maxlen=args["listSize"])
#kernel function
def MakeKernel( x ):
    total = cv2.circle(np.zeros((x,x), np.uint8),(int(x/2),int(x/2)),int(x/2) ,1,-1);
    return total

#euclidean distance function
def euclidean(vector1, vector2):
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist

#dirs to save
dip_img = "./dataset/images/"
dip_txt = "./dataset/Labelbox/"
kkk = "./dataset/kkk/"
last_txt = os.listdir( dip_txt )[-1]
num = int(last_txt.split('-')[1].split('.')[0])
#getting cam data
vs = webcamVideoStreamFPS(src=0,nFrames=int(args["num_frames"])).start()
fps = FPS().start()

key = cv2.waitKey(1) & 0xFF
while key != ord('q'):

    frame = vs.read()
    frame = imutils.resize(frame, height=args["width"])
    orig  = frame

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    thresh = threshold_otsu(gray)

    ret,imgThresh = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)

    img_erosion = cv2.erode(imgThresh, MakeKernel(5), iterations=1)
    img_dilation = cv2.dilate(img_erosion, MakeKernel(10), iterations=1)

    #get structure data foreach garlic
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_dilation)

    height,width,channels = frame.shape
    num = num+1
    name = 'image-'+'0'*(7-len(str(num)))+str(num)
    cv2.imwrite(kkk+name+'.png',frame)
    file = open(kkk+name+'.txt','w')
    ##[<label>,<x>,<y>,<width>,<height>]
    #drawing bounding box
    for pos in range(len(stats)):
        stat = stats[pos]
        cen = centroids[pos]
        if stat[cv2.CC_STAT_AREA] <= 90000  and stat[cv2.CC_STAT_AREA] >= 10000:
            x = stat[cv2.CC_STAT_LEFT]  / width
            y = stat[cv2.CC_STAT_TOP]   / height
            w = stat[cv2.CC_STAT_WIDTH] / width
            h = stat[cv2.CC_STAT_HEIGHT]/ height
            file.write('0'+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n')

    cv2.imshow('image',frame)
    file.close()
    key = cv2.waitKey(1) & 0xFF
    fps.update()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

