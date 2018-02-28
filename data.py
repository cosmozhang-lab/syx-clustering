import numpy as np

class RandomBatchIndexes:
  def __init__(self, datasize):
    self.datasize = datasize
    self.indexes = np.arange(datasize)
    np.random.shuffle(self.indexes)
    self.start = 0
  def next(self, batch_size):
    assert not batch_size > self.datasize
    start = self.start
    nextstart = (start + batch_size) % self.datasize
    self.start = nextstart
    if nextstart > start:
      return self.indexes[start:nextstart]
    else:
      return np.concatenate((self.indexes[start:], self.indexes[:nextstart]))
