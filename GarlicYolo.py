from __future__ import print_function
import math
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import LearningRateScheduler,EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
import numpy as np
import pickle, random
import os, cv2, sys
from utils import BoundBox, bbox_iou
from preprocessing import parse_annotation,BatchGenerator
from utils import WeightReader, decode_netout, draw_boxes, _sigmoid, BoundBox
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from utils import *

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

generator_config = {
    'IMAGE_H'         : IMAGE_H, 
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'CLASS'           : len(LABELS),
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}

def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)

def normalize(image):
    return image / 255.


wt_path            = 'weights_garlic.h5'
train_image_folder = './dataset/images/'
train_annot_folder = './dataset/annots/'

train_imgs, seen_labels = parse_annotation(train_annot_folder, train_image_folder, labels=LABELS)
random.shuffle(train_imgs)

x_train ,x_test = train_test_split(train_imgs,test_size=0.2)
x_train ,x_val = train_test_split(x_train,test_size=0.1)

train_batch = BatchGenerator(x_train, generator_config, norm=normalize)
test_batch  = BatchGenerator(x_test, generator_config, norm=normalize)
valid_batch = BatchGenerator(x_val, generator_config, norm=normalize)

input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

# Layer 1
x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 2 - 5
for i in range(0,4):
    x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
    x = BatchNormalization(name='norm_' + str(i+2))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

# Layer 7 - 8
for i in range(0,3):
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
    x = BatchNormalization(name='norm_' + str(i+7))(x)
    x = LeakyReLU(alpha=0.1)(x)

x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)

output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)
output = Lambda(lambda args: args[0])([output, true_boxes])


model = Model([input_image, true_boxes], output)  
model.summary()

# ##adding pretrained yolo weigth
# weight_reader = WeightReader("./yolov2.weights")
# weight_reader.reset()
# nb_conv = 9

# for i in range(1, nb_conv+1):
#     conv_layer = model.get_layer('conv_' + str(i))
#     if i < nb_conv:
#         norm_layer = model.get_layer('norm_' + str(i))
#         size = np.prod(norm_layer.get_weights()[0].shape)
#         beta  = weight_reader.read_bytes(size)
#         gamma = weight_reader.read_bytes(size)
#         mean  = weight_reader.read_bytes(size)
#         var   = weight_reader.read_bytes(size)
#         weights = norm_layer.set_weights([gamma, beta, mean, var])       
#     if len(conv_layer.get_weights()) > 1:
#         bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
#         kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
#         kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
#         kernel = kernel.transpose([2,3,1,0])
#         conv_layer.set_weights([kernel, bias])
#     else:
#         kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
#         kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
#         kernel = kernel.transpose([2,3,1,0])
#         conv_layer.set_weights([kernel])


        
# layer   = model.layers[-6] # the last convolutional layer
# weights = layer.get_weights()
# new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
# layer.set_weights([new_kernel])


# ##Load pretrained weights
exists = os.path.isfile(wt_path)
if exists:
    model.load_weights(wt_path)

##Loss function
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


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > EPOCHS*0.8:
        lr *= 0.5e-3
    elif epoch > EPOCHS*0.6:
        lr *= 1e-3
    elif epoch > EPOCHS*0.4:
        lr *= 1e-2
    elif epoch > EPOCHS*0.2:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


#optimizer
optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0005)
# early stop
early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=5,mode='min',verbose=1)
# checkpoint
checkpoint = ModelCheckpoint(
    os.path.join('checkpoints','weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min')
# DEFINE A LEARNING RATE AND REDUCTION SCHEDULER
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',patience=2,verbose=1, factor=0.5,min_lr=0.00001)
Lr_reduction = LearningRateScheduler(lr_schedule)

model.compile(loss=custom_loss, optimizer=optimizer)


history=model.fit_generator(generator        = train_batch, 
                            steps_per_epoch  = len(train_batch), 
                            epochs           = EPOCHS,
                            validation_data  = valid_batch,
                            validation_steps = len(valid_batch),
                            callbacks        = [learning_rate_reduction,checkpoint,Lr_reduction],
                            verbose          = 1)

model.save('GarlicYolo.hdf5')

score = 0
num_boxes=0
print('Evaluating network...')
for i in tqdm(x_test):
    name = i['filename']
    imagen = np.expand_dims(cv2.resize(cv2.imread(name),(int(416),int(416))),0)
    out = np.squeeze(imagen)
    imagen = imagen/255
    netout = model.predict([imagen, np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))])
    boxes,iou = decode_netout(netout[0], obj_threshold=OBJ_THRESHOLD,nms_threshold=NMS_THRESHOLD,anchors=ANCHORS,nb_class=CLASS)
    for v in i['object']:
        image_box = BoundBox(v['xmin']/IMAGE_W,v['ymin']/IMAGE_H,v['xmax']/IMAGE_W,v['ymax']/IMAGE_H,classes=LABELS)
        out = draw_boxes(out,[image_box], labels=LABELS)
        for b in boxes:
            b.classes=LABELS
            out = draw_boxes(out,[b], labels=LABELS,colorB=(255,0,0))
            iou = bbox_iou(image_box, b)
            score = score + iou
            num_boxes = num_boxes + 1

print('IOU test: {:2f}%'.format((1-(score/num_boxes))*100))

#OUTPUT
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
