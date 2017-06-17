
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import pdb
import random as rand

import numpy as np
from six.moves import urllib
import tensorflow as tf

import data_proc

FLAGS = tf.app.flags.FLAGS

SCALE = 0.0 #not relevant here
name = 'reluDecayLinearQuick'
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of images to process in a batch.""")

home_dir = os.getenv('HOME')

DIM = [data_proc.COMPR_SIZE[1], data_proc.COMPR_SIZE[0], 9]

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.0     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 128.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.
WEIGHT_DECAY = 0.001
TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.
  
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  
  Args:
    x: Tensor
  Returns:
  nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
  
def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var
def batchnorm(input, suffix, is_train):
  rank = len(input.get_shape().as_list())
  in_dim = input.get_shape().as_list()[-1]

  if rank == 2:
    axes = [0]
  elif rank == 4:
    axes = [0, 1, 2]
  else:
    raise ValueError('Input tensor must have rank 2 or 4.')

  mean, variance = tf.nn.moments(input, axes)
  offset = _variable_on_cpu('offset_' + str(suffix), in_dim,
      tf.constant_initializer(0.0))
  scale = _variable_on_cpu('scale_' + str(suffix), in_dim, 
    tf.constant_initializer(1.0))
  # offset = 0.0
  #scale = 1.0
  
  decay = 0.95
  epsilon = 1e-4

  ema = tf.train.ExponentialMovingAverage(decay=decay)
  ema_apply_op = ema.apply([mean, variance])
  ema_mean, ema_var = ema.average(mean), ema.average(variance)

  if is_train:
    with tf.control_dependencies([ema_apply_op]):
      return tf.nn.batch_normalization(
              input, mean, variance, offset, scale, epsilon)
  else:
      # batch = tf.cast(tf.shape(x)[0], tf.float32)
      # mean, var = ema_mean, ema_var * batch / (batch - 1) # unbiased variance estimator
    return tf.nn.batch_normalization(
      input, ema_mean, ema_var, offset, scale, epsilon)


def inference(images, is_train):
# biases_btst = _variable_on_cpu('biases_btst', [64, DIM[2]], tf.constant_initializer(0.0))
# biases_btst2 = _variable_on_cpu('biases_btst2', [48, 64, DIM[2]], tf.constant_initializer(0.0))
# images = tf.add(images, biases_btst)
# images = tf.add(images, biases_btst2)

  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
		shape=[4, 4, DIM[2], 32],
		stddev=5e-2,
		wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    conv1 = batchnorm(conv1, "conv1_bn", is_train)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
	    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
	                                 shape=[6, 6, 32, 32],
	                                 stddev=5e-2,
	                                 wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    conv2 = batchnorm(conv2, "conv2_bn", is_train)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
	    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
	    strides=[1, 2, 2, 1], padding='SAME', name='pool2')


  # local3
  with tf.variable_scope('local3') as scope:
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    #assert(dim == 12 * 16 * 32)
    #print(dim)
    weights = _variable_with_weight_decay('weights', shape=[32 * (DIM[0] / 4) * (DIM[1] / 4), 384], stddev=0.04, wd=WEIGHT_DECAY )
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  with tf.variable_scope('linear_end') as scope:
    local4 = local3
    weights = _variable_with_weight_decay('weights', shape=[384, 1], stddev=0.04, wd=WEIGHT_DECAY)
    biases = _variable_on_cpu('biases', [1], tf.constant_initializer(5.0))
    linear_end = tf.matmul(local4, weights) + biases
    _activation_summary(linear_end)
    epsilon = 1e-4

  return linear_end

def loss(ans, labels):
  tf.add_to_collection('losses', tf.reduce_mean(tf.abs(ans - labels), name = 'squared_diff'))
  return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

def loss1(ans, labels):
  return tf.reduce_mean(tf.abs(ans - labels), name = 'squared_diff')

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def train(total_loss, global_step):

  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  print('num batch per epoch', num_batches_per_epoch, 'decay steps', decay_steps)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  
  tf.summary.scalar('learning_rate', lr)
  #tf.summary.scalar('reluVal', modifiedRelu(-1.0))

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.MomentumOptimizer(lr, 0.9)
    # opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def ram_inputs(data_dir, is_train):
  return data_proc.ram_inputs(data_dir, is_train)

class BatchGenerator(object):

  def __init__(self, images, labels, is_train, num_epochs):
    self.images = images
    self.labels = labels
    self.is_train = is_train
    self.batch_size = FLAGS.batch_size
    self.num_epochs = num_epochs
    self.curr_ind = 0
    self.curr_epoch = 0
    self.done = False
    self._num_samples = images.shape[0]
    # Assume square image.
    self._imsize = images.shape[1]

    if is_train:
      self.shuffle()


  def is_done(self):
    return self.done

  def num_samples(self):
    return self._num_samples

  def shuffle(self):
    perm = np.arange(self.images.shape[0])
    np.random.shuffle(perm)
    self.images = self.images[perm]
    self.labels = self.labels[perm]

  def next_batch(self):
    end_ind = self.curr_ind + self.batch_size
    do_shuffle = False
    if end_ind >= self.images.shape[0]:
      # Epoch finished
      end_ind = self.images.shape[0]
      self.curr_epoch = self.curr_epoch + 1
      if self.curr_epoch == self.num_epochs:
        self.done = True
      # Shuffle the dataset
      do_shuffle = True

    indices = slice(self.curr_ind, end_ind)
    #x = [make_train_input(self.images, self.)]
    images = self.images[indices]
    labels = self.labels[indices]

    if end_ind == self.images.shape[0]:
      self.curr_ind = 0
    else:
      self.curr_ind = end_ind

    # Do the augmentation
    if self.is_train:
      # LR flip.
      if np.random.rand(1) > 0.5:
        images = images[:,:,::-1,:]

    if do_shuffle:
      self.shuffle()

    return images, labels

