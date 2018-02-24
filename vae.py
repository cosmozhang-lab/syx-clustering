import tensorflow as tf
import numpy as np

class VaeModel:
  def __init__(self):
    self.input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    self.label = tf.placeholder(tf.float32, shape=[None, 10])
    self.inputflat = tf.reshape(self.input, [-1, 28 * 28])
    # Construct the encoder
    x1 = self.input
    w1 = tf.Variable(tf.truncated_normal([28 * 28, 128], stddev=0.1))
    b1 = tf.Variable(tf.constant(0, [128]))
    y1 = tf.nn.relu(tf.matmul(x1, w1) + b1)
    x2 = y1
    w2m = tf.Variable(tf.truncated_normal([128, 10]))
    b2m = tf.Variable(tf.constant(0, [10]))
    y2m = tf.nn.relu(tf.matmul(x2, w2m) + b2m)
    w2s = tf.Variable(tf.truncated_normal([128, 10]))
    b2s = tf.Variable(tf.constant(0, [10]))
    y2s = tf.nn.relu(tf.matmul(x2, w2s) + b2s)
    x3m = y2m
    x3s = y2s
    y3 = x3m + tf.matmul(x3s, tf.truncated_normal([10]))
    self.encoder = y3
    # Construct the decoder
    x4 = y3
    w4 = tf.Variable(tf.truncated_normal([10, 128], stddev=0.1))
    b4 = tf.Variable(tf.constant(0, [128]))
    y4 = tf.nn.relu(tf.matmul(x4, w4) + b4)
    x5 = y4
    w5 = tf.Variable(tf.truncated_normal([128, 28 * 28], stddev=0.1))
    b5 = tf.Variable(tf.constant(0, [28 * 28]))
    y5 = tf.nn.relu(tf.matmul(x5, w5) + b5)
    y5reshaped = tf.reshape(y5, [-1, 28, 28, 1])
    self.decoder = y5reshaped
    # Loss
    self.loss = tf.reduce_mean()

