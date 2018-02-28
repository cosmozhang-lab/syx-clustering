import numpy as np
from vae import VaeModel
from data import RandomBatchIndexes
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

randombatch = RandomBatchIndexes(mnist.train.labels.shape[0])
model = VaeModel()
model.restore("saving/train.ckpt")
batchidxes = randombatch.next(100)
images = np.reshape(mnist.train.images[batchidxes], [-1, 28, 28, 1])
labels = mnist.train.labels[batchidxes]
generated = model.sess.run(model.decoder, feed_dict = { model.input: images })

panel_origin = np.zeros((28 + 4) * 10 - 4, (28 + 4) * 10 - 4)
panel_generated = np.zeros((28 + 4) * 10 - 4, (28 + 4) * 10 - 4)
