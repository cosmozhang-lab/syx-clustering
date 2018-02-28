import tensorflow as tf
import numpy as np

class VaeModel:
  def __init__(self):
    self.input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    # self.label = tf.placeholder(tf.float32, shape=[None, 10])
    self.inputflat = tf.reshape(self.input, [-1, 28 * 28])
    # Construct the encoder
    x1 = tf.reshape(self.input, [-1, 28 * 28])
    w1 = tf.Variable(tf.truncated_normal([28 * 28, 128], stddev=0.1))
    b1 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[128]))
    y1 = tf.nn.relu(tf.matmul(x1, w1) + b1)
    x2 = y1
    w2m = tf.Variable(tf.truncated_normal([128, 10]))
    b2m = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[10]))
    y2m = tf.nn.relu(tf.matmul(x2, w2m) + b2m)
    w2s = tf.Variable(tf.truncated_normal([128, 10]))
    b2s = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[10]))
    y2s = tf.nn.relu(tf.matmul(x2, w2s) + b2s)
    x3m = y2m
    x3s = y2s
    y3 = x3m + tf.matmul(x3s, tf.truncated_normal([10,1]))
    self.encoder = y3
    # Construct the decoder
    x4 = y2m
    w4 = tf.Variable(tf.truncated_normal([10, 128], stddev=0.1))
    b4 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[128]))
    y4 = tf.nn.relu(tf.matmul(x4, w4) + b4)
    x5 = y4
    w5 = tf.Variable(tf.truncated_normal([128, 28 * 28], stddev=0.1))
    b5 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[28 * 28]))
    y5 = tf.nn.relu(tf.matmul(x5, w5) + b5)
    y5reshaped = tf.reshape(y5, [-1, 28, 28, 1])
    self.decoder = y5reshaped
    # Loss
    self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(y5 - x1), 1), 0)
    # Runner
    self.optimizer = tf.train.GradientDescentOptimizer(0.001)
    self.train_step = self.optimizer.minimize(self.loss)
    self.sess = tf.Session()

  def initialize(self):
    self.sess.run(tf.initialize_all_variables())

  def train(self, images):
    images = np.reshape(images, [-1, 28, 28, 1])
    self.sess.run(self.train_step, feed_dict = { self.input: images })
    return self.sess.run(self.loss, feed_dict = { self.input: images })

  def save(self, filename):
    saver = tf.train.Saver()
    saver.save(self.sess, filename)
  def restore(self, filename):
    saver = tf.train.Saver()
    saver.restore(self.sess, filename)









