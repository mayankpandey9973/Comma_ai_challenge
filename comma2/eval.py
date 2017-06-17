import tensorflow as tf
import numpy as np
import nnet
import data_proc

images = data_proc.extract_images("raw_train", 20000)
labels = data_proc.extract_labels("train.txt")
mean_images = np.mean(images[0:data_proc.NUM_TRAIN_IMAGES+11], axis = 0)
sqrt_var = np.std(images[0:data_proc.NUM_TRAIN_IMAGES+11])
valid_images = np.asarray([data_proc.make_train_input(images, labels, index, mean_images, sqrt_var, normalize = True) for index in range(2000)])
lab_avgs = np.asarray([np.mean(labels[index:index+3]) for index in range(2000)])



valid_images = valid_images.astype('float32')
for i in range(valid_images.shape[0] / 100):
    with tf.Graph().as_default() as g:
	otpts = nnet.inference(valid_images[100 * i: 100 * (i + 1)], is_train = False)
	variable_averages = tf.train.ExponentialMovingAverage(0.99999)
	variables_to_restore = variable_averages.variables_to_restore()
	saver = tf.train.Saver(variables_to_restore)
	
	with tf.Session() as sess:
	    ckpt = tf.train.get_checkpoint_state('/home/mayank/comma_ai_logs/train_data')
	    #print(sess.run(variables_to_restore))
	    #print(sess.run(ckpt))
	    saver.restore(sess, ckpt.model_checkpoint_path)
	    predicts = sess.run([otpts])
	    print("Precision: " + str(np.mean(np.square(predicts - lab_avgs[100 * i: 100 * (i + 1)]))))
