#!/usr/bin/env python3
from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
import numpy as np
import pickle
import os, cv2
from keras.models import load_model
from preprocessing import parse_annotation, BatchGenerator
from utils import WeightReader, decode_netout, draw_boxes
import keras.losses

from WebcamVideoStreamFPS import *

import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument("-i", "--image",
    default='./dataset/images/',
    help="path to input images")
argparser.add_argument("-v", "--video",
    default='./garlic.mp4',
    help="path to input video")
argparser.add_argument("-m", "--model",
    default='weights_garlic.h5',
    help="path to trained model file")
argparser.add_argument("-w", "--weights",
    default="GarlicYolo.hdf5",
    help="path to trained weights file")
argparser.add_argument("-xm", "--mode",
    default=-1,
    help="Choose False for video or True for image")
args = argparser.parse_args()

LABELS = ['garlic']

IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13,13
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.75
NMS_THRESHOLD    = 0.1
ANCHORS          = [3.55,14.54, 4.81,7.50, 5.90,10.56, 7.19,5.84, 8.71,8.41]
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 20
WARM_UP_BATCHES  = 1
TRUE_BOX_BUFFER  = 10
EPOCHS           = 100

def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)


def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:4]
    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)),tf.float32)
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])
    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)
    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)
    """
    Adjust prediction
    """
    ### adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    ### adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    ### adjust class probabilities
    pred_box_class = y_pred[..., 5:]
    """
    Adjust ground truth
    """
    ### adjust x and y
    true_box_xy = y_true[..., 0:2] # relative position to the containing cell
    ### adjust w and h
    true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
    ### adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins    = true_box_xy - true_wh_half
    true_maxes   = true_box_xy + true_wh_half
    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxes   = pred_box_xy + pred_wh_half   
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]  
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)
    true_box_conf = iou_scores * y_true[..., 4]
    ### adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
    ### confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)
    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)  
    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.cast((best_ious < 0.6),tf.float32) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE
    ### class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE       
    """
    Warm-up training
    """
    no_boxes_mask = tf.cast((coord_mask < COORD_SCALE/2.),tf.float32)
    seen = tf.assign_add(seen, 1.)
    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES), 
                          lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                   true_box_wh + tf.ones_like(true_box_wh) * np.reshape(ANCHORS, [1,1,1,BOX,2]) * no_boxes_mask, 
                                   tf.ones_like(coord_mask)],
                          lambda: [true_box_xy, 
                                   true_box_wh,
                                   coord_mask])
    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.cast((coord_mask > 0.0),tf.float32))
    nb_conf_box  = tf.reduce_sum(tf.cast((conf_mask  > 0.0),tf.float32))
    nb_class_box = tf.reduce_sum(tf.cast((class_mask > 0.0),tf.float32))
    loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    loss = loss_xy + loss_wh + loss_conf + loss_class
    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(tf.cast((true_box_conf > 0.5),tf.float32) * tf.cast((pred_box_conf > 0.3),tf.float32))
    """
    Debugging code
    """    
    current_recall = nb_pred_box/(nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall)  
    # loss = tf.Print(loss, [], message='\t', summarize=1000)
    # loss = tf.Print(loss, [], message='********************\t', summarize=1000)
    # loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    # loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    # loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    # loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    # loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    # loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    # loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
    # loss = tf.Print(loss, [], message='********************\t', summarize=1000) 
    return loss

true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))
model = load_model(args.model, custom_objects={'custom_loss': custom_loss,'true_boxes' : true_boxes})
# ##Load pretrained weights
model.load_weights(args.weights)

print("------------- MODE: {}".format(args.mode))

if(int(args.mode) == 0):
    ##Single image detector
    iou_l = []
    for file in os.listdir(args.image):
        image = cv2.imread(args.image+file)
        dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
        plt.figure(figsize=(10,10))
        input_image = cv2.resize(image, (IMAGE_H, IMAGE_W))
        input_image = input_image / 255.
        input_image = input_image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        netout = model.predict([input_image, dummy_array])
        boxes,iou = decode_netout(netout[0], obj_threshold=OBJ_THRESHOLD,nms_threshold=NMS_THRESHOLD,anchors=ANCHORS,nb_class=CLASS)
        iou_l.append(iou)
        if(True):
            image = draw_boxes(image, boxes, labels=LABELS)
            plt.imshow(image[:,:,::-1]); plt.show()
            print("Average IOU: {:2f}".format(iou))

    print("Average IOU: {:2f}".format(np.mean(iou_l)))
elif(int(args.mode) == 1):
    ##Video Detector
    print("Loading video...")
    video_inp = args.video
    video_reader = cv2.VideoCapture(video_inp)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_writer = cv2.VideoWriter("outputVideo.mp4",-1,50.0,(frame_w, frame_h))
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
    for i in tqdm(range(nb_frames)):
        ret, image = video_reader.read()
        input_image = cv2.resize(image, (IMAGE_H, IMAGE_W))
        input_image = input_image / 255.
        input_image = input_image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        netout = model.predict([input_image, dummy_array])
        boxes = decode_netout(netout[0], 
                              obj_threshold=OBJ_THRESHOLD,
                              nms_threshold=NMS_THRESHOLD,
                              anchors=ANCHORS, 
                              nb_class=CLASS)
        image = draw_boxes(image, boxes[0], labels=LABELS)
        video_writer.write(np.uint8(image))
    video_reader.release()
    video_writer.release()

else:
    vs = webcamVideoStreamFPS(src=0,nFrames=30).start()
    key = cv2.waitKey(1) & 0xFF
    while key != ord('q'):
        frame = vs.read()
        dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
        input_image = cv2.resize(frame,(IMAGE_H,IMAGE_W))
        input_image = input_image / 255.
        input_image = input_image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        netout = model.predict([input_image, dummy_array])
        boxes = decode_netout(netout[0], 
                              obj_threshold=OBJ_THRESHOLD,
                              nms_threshold=NMS_THRESHOLD,
                              anchors=ANCHORS, 
                              nb_class=CLASS)
        image = draw_boxes(frame, boxes[0], labels=LABELS)
        
        cv2.imshow('Live',image)
        key = cv2.waitKey(1) & 0xFF

    cv2.destroyAllWindows()
    vs.stop()