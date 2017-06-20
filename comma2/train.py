from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import nnet
import data_proc
# from tensorflow.models.image.cifar10 import cifar10

n = 5
FLAGS = tf.app.flags.FLAGS
home_dir = os.getenv('HOME')

tf.app.flags.DEFINE_string('train_dir',
	home_dir + '/comma_ai_logs/train_data_small',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_epochs', 64*2+32 + 1,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():

    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    # images, labels = cifar10.standard_distorted_inputs()
    inputs = nnet.ram_inputs(data_dir = 'raw_train', is_train=True)
    images = inputs['images']
    labels = inputs['labels']

    # Batch generator
    batcher = nnet.BatchGenerator(
        inputs['data_images'], inputs['data_labels'], True,
        FLAGS.max_epochs)
    print("Batcher made")

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = nnet.inference(images, is_train=True)

    # Calculate loss.
    loss = nnet.loss(logits, labels)
    loss1 = nnet.loss1(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = nnet.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    if False:
      sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=gpu_options))
    else:
      sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))

    sess.run(init)


    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    print("bailing out here at 1")

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    step = -1
    while not batcher.is_done():
      step += 1

      batch_im, batch_labs = batcher.next_batch()
      feed_dict = {
          inputs['images_pl']: batch_im,
          inputs['labels_pl']: batch_labs,
      }
      #print(sess.run(tf.shape(batch_im)))

      start_time = time.time()
      _, loss_value, loss1_value = sess.run([train_op, loss, loss1], feed_dict=feed_dict)

      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

              
      if step % 1 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f, sdiff = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch), writing to ' + FLAGS.train_dir + '.')
        print (format_str % (datetime.now(), step, loss_value,
                             loss1_value, examples_per_sec, sec_per_batch))

      if step % 10 == 0:
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 100 == 0 or batcher.is_done():
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step % 300 == 0:
	print("Here are some labels: %s" % str(batch_labs))


        
def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()


