#This file just extracts the test data, and has some functions for returning batches
#
#Intended to have essentially the same format as CIFAR-10 demo files
#
#
#

import os
import glob
import io
import numpy as np
import pylab
import imageio
import matplotlib.image as mpimg
import tensorflow as tf
from PIL import Image
from natsort import natsorted

ROWS = 482
COLS = 640
COMPR_SIZE = (64 * 2, 48 * 2)
#NUM_TRAIN_IMAGES = 20000
NUM_TRAIN_IMAGES = 5000

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

def make_train_input(images, labels, index, mean_images, sqrt_var, normalize = True):
    frame_used = (images[index:index + 3] - (mean_images if normalize else 0.0))/(sqrt_var if normalize else 1.0)
    #print(np.shape(frame_used))
    frame_glue = np.zeros((COMPR_SIZE[1], COMPR_SIZE[0], 3*3))
    #print("to compute axis 2 means")
    for i in range(3):
        frame_glue[:,:,i*3:i*3+3] = frame_used[i]
    
    return frame_glue

def ram_inputs(data_dir, is_train):
   
    print("Gluing everything")
    images = extract_images("raw_train", NUM_TRAIN_IMAGES)[0:NUM_TRAIN_IMAGES + 11]
    mean_images = np.mean(images, axis = 0)
    sqrt_var = np.std(images, 0)
    labels = extract_labels("train.txt")
    print("Label len type =", len(labels), type(labels[1]), labels[:20])
    data_images = np.asarray([make_train_input(images, labels, index, mean_images, sqrt_var, normalize = is_train) for index in range(NUM_TRAIN_IMAGES)])

    data_labels = np.asarray([np.mean(labels[index:index + 10]) for index in range(NUM_TRAIN_IMAGES)])
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




