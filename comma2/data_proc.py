#This file just extracts the test data, and has some functions for returning batches
#
#Intended to have essentially the same format as CIFAR-10 demo files
#
#
#

import os
import glob
import io
import cv2
import numpy as np
import pylab
import imageio
import matplotlib.image as mpimg
import tensorflow as tf
from PIL import Image
from natsort import natsorted

ROWS = 482
COLS = 640
COMPR_SIZE = (64, 48)
NUM_TRAIN_IMAGES = 15000
#NUM_TRAIN_IMAGES = 20400

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

#gives images, labels 
def extract_images_raw(folder_name, num_images):
#print(folder_name)
    dir_name = "%s/%s/" % ("data", folder_name)
    lst = natsorted(listdir_nohidden(dir_name))
#print(lst[0], lst[1])
    im_array = np.array([np.array(Image.open(lst[i]).resize(COMPR_SIZE, Image.ANTIALIAS)) for i in range(num_images)])
    print("have image array")
    print(np.shape(im_array))
    i = 0
    #    for name in lst:
    #	img = mpimg.imread('%s/%s' % (dir_name, name))
    #	num = int(name[5:str.find(name, '.')]) - 1
    #	images[num] = img
    #	i += 1
    #	if i % 100 == 0: 
    #	    print("have read: " + str(i))

    return im_array

def extract_images(folder_name, num_images, grayscale=True):
    file_name = "data/" + folder_name + ".npy"
    if os.path.exists(file_name):
        print("Have log of previous files, loading that")
        images = np.load(file_name)
        # ?x48x64x3
#        if grayscale:
#            images = np.mean(images, axis = 3)
        return images
    else:
        images = extract_images_raw(folder_name, num_images)
        np.save(file_name, images)
        return images
	    

def extract_labels(fname):
    print("label extract starting")
    lab_name = "data/" + fname
    labels = []
    infile = open(lab_name, "r")
    labels = np.array([float(line) for line in infile])
    print("label extract done")
    return np.array(labels)

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def make_train_input(images, labels, index, mean_images, sqrt_var, normalize = False):
    frame_used = images[index:index + 3]
#print(np.shape(frame_used))
    flows = [opticalFlowDense(frame_used[i], frame_used[i + 1]) for i in range(3 - 1)]
    frame_glue = np.zeros((COMPR_SIZE[1], COMPR_SIZE[0], (3 - 1)*3))
    #print("to compute axis 2 means")
    for i in range(2):
        frame_glue[:,:,i*3:i*3+3] = flows[i]
    
    return (frame_glue - np.mean(frame_glue)) / np.std(frame_glue)
def opticalFlowDense(image_current_arr, image_next_arr):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    image_current = image_current_arr #Image.fromarray(image_current_arr, 'RGB')
    image_next = image_next_arr #Image.fromarray(image_next_arr, 'RGB')
    
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
    
    
    hsv = np.zeros((COMPR_SIZE[1], COMPR_SIZE[0], 3))
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]
 
    # Flow Parameters
    #     flow_mat = cv2.CV_32FC2
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,  
                                        flow_mat, 
                                        image_scale, 
                                        nb_images, 
                                        win_size, 
                                        nb_iterations, 
                                        deg_expansion, 
                                        STD, 
                                        0)
                                        
        
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  
        
    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return np.asarray(rgb_flow)

def ram_inputs(data_dir, is_train):
   
    print("Gluing everything")
    images = extract_images("raw_train", NUM_TRAIN_IMAGES)[0:NUM_TRAIN_IMAGES + 11]
    mean_images = np.mean(images, axis = 0)
    sqrt_var = np.std(images, 0)
    labels = extract_labels("train.txt")
    print("Label len type =", len(labels), type(labels[1]), labels[:20])
    data_images = np.asarray([make_train_input(images, labels, index, mean_images, sqrt_var, normalize = is_train) for index in range(NUM_TRAIN_IMAGES)])

    data_labels = np.asarray([np.mean(labels[index:index + 3]) for index in range(NUM_TRAIN_IMAGES)])
    shuffle_in_unison(data_images, data_labels)
    
    print("data_labels", data_labels[:20])
    print("done gluing")
    print("total size: " + str(data_images.size + images.size + data_labels.size))
    height = COMPR_SIZE[1]
    width = COMPR_SIZE[0]

    with tf.name_scope('input'):
        # Input data
        images_pl = tf.placeholder(dtype=data_images.dtype, shape=[None, data_images.shape[1], data_images.shape[2], data_images.shape[3]])
        labels_pl = tf.placeholder(dtype=data_labels.dtype, shape=[None])    
        images = tf.cast(images_pl, tf.float32)
        labels = tf.cast(labels_pl, tf.float32)

    ret_val = {
	'images': images,
	'labels': labels,
	'images_pl': images_pl,
	'labels_pl': labels_pl,
	'data_images': data_images,
	'data_labels': data_labels,
	'mean_images': mean_images
	'sqrt_var': sqrt_var
    }

    return ret_val

#def 


#inputs = ram_inputs("raw_train", is_train = False)

#images = extract_images("raw_train", NUM_TRAIN_IMAGES)
#labels = extract_labels("train.txt")
#print(labels[0:100])
#print(np.shape(labels))
#print(labels[0:10])
#glued = [make_train_input(images, labels, 0)[0][:,:,i] for i in range(10)]
#img = [Image.fromarray(glued[i]) for i in range(10)]
#[img[i].show() for i in range(10)]




