import time 
import os
import io
import numpy as np
import pylab
import imageio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import data_proc

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from natsort import natsorted

ROWS = 482
COLS = 640
COMPR_SIZE = (64, 48)
shift_size = 5
gran =  25
numSects = int(256 / gran + 1)
NUM_IMAGES = 20000

def addFreq(img, cnts, totals):
    for index in np.ndindex(48, 64):
        col = img[index]
        totals[int(col[0] / gran)][int(col[1] / gran)][int(col[2] / gran)] += index
        cnts[int(col[0] / gran)][int(col[1] / gran)][int(col[2] / gran)] += 1

def addFreqs(images, s, e, cnts, totals):
    for i in range(s, e):
        if i % 100 == 0: print("Have added img " + str(i)) 
        addFreq(images[i], cnts[i], totals[i])

images = data_proc.extract_images('raw_train', 20000, False)

file_name_tots = "data/colrTots.npy"
file_name_cnts = "data/colrCnts.npy"
file_name_avgs = "data/colrAvgs.npy"


colrCnts = []
locTots = []
avgColrs = []
if os.path.exists(file_name_avgs):
    start = time.clock()
    avgColrs = np.load(file_name_avgs)
    colrCnts = np.load(file_name_cnts)
    locTots = np.load(file_name_tots)
    print("Loaded avgs in " + str(time.clock() - start))
else:
    if os.path.exists(file_name_cnts) and os.path.exists(file_name_tots):
        start = time.clock()
        colrCnts = np.load(file_name_cnts)
        locTots = np.load(file_name_tots)

        print("Loaded in " + str(time.clock() - start))
    else:
        colrCnts = np.zeros((20000, numSects, numSects, numSects, 1))
        locTots = np.zeros((20000, numSects, numSects, numSects, 2))
        #    for index, img in enumerate(images[:NUM_IMAGES]):
        #	if index%100 == 0: print(index)
        #	addFreq(img, colrCnts, locTots)
        start = time.clock()
        addFreqs(images, 0, NUM_IMAGES, colrCnts, locTots)
        print("Took " + str(time.clock() - start))
        print(np.sum(colrCnts))
        print(np.size(colrCnts) + np.size(locTots))
        #exit(0)
        np.save(file_name_cnts, colrCnts)
        np.save(file_name_tots, locTots)
    avgColrs = locTots / colrCnts
    np.save(file_name_avgs, avgColrs)

labels = data_proc.extract_labels("train.txt")
NUM_SAMPLE = 20000
med_scatter = []
mea_scatter = []
vel_scatter = []
for index in range(NUM_SAMPLE - 5):
    im_dat = np.asarray([np.reshape(avgColrs[index + i], (numSects * numSects * numSects, 2)) for i in range(6)])
    diff = np.zeros(np.shape(im_dat[0]))
    for im1 in im_dat:
        for im2 in im_dat:
            diff += (im1 - im2) * (im1 - im2)
    dist = diff[:,0] + diff[:,1]

   
    cleaned_dist = np.asarray([x for x in dist if str(np.sum(x)) != 'nan'])
    #    clean_im1 = np.asarray([x for x in im1_dat if str(np.sum(x)) != 'nan'])
    #    clean_im2 = np.asarray([x for x in im2_dat if str(np.sum(x)) != 'nan'])
    #    clean_sum = np.asarray([x for x in im2_dat + im1_dat if str(np.sum(x)) != 'nan'])
    med_scatter.append([np.median(cleaned_dist), labels[index]])
    mea_scatter.append([np.mean(cleaned_dist), labels[index]])

#if str(np.median(cleaned_dist)) != 'nan': print(str(index) + ": " + str(np.median(cleaned_dist)) + ", " + str(np.mean(cleaned_dist)))

#    print(cleaned_dist.shape[0], clean_im1.shape[0], clean_im2.shape[0], clean_sum.shape[0])
#    print(np.sum(colrCnts[index]))



med_scatter = np.asarray(med_scatter)
mea_scatter = np.asarray(mea_scatter)

plt.plot(med_scatter[:,0], med_scatter[:,1])
plt.plot(mea_scatter[:,0], mea_scatter[:,1])
plt.show()

#for






#for index in range(NUM_IMAGES):

	
