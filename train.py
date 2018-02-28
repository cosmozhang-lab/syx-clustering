import numpy as np
from vae import VaeModel
from data import RandomBatchIndexes
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

randombatch = RandomBatchIndexes(mnist.train.labels.shape[0])
model = VaeModel()
model.initialize()
for i in xrange(1000):
  batchidxes = randombatch.next(128)
  loss = model.train(mnist.train.images[batchidxes])
  print "#%03d Loss: %f" % (i, loss)
# print model.sess.run(model.loss, feed_dict = { model.input: np.reshape(mnist.train.images[randombatch.next(128)], [-1, 28, 28, 1]) })
model.save("saving/train.ckpt")