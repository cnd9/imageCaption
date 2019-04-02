# -*- coding: utf-8 -*-
"""
Created on Mon May  9 22:28:28 2016

@author: ubuntu
"""

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
#from skimage.draw import polygon

from collections import defaultdict

class WORDVECS:
  def __init__(self,vocab):
      self.process_sentences()
      self.embeddings = self.construct_embeddings(vocab)

  def process_sentences(self):
    wordvecdict = {}
    with open('./glove.6B.200d.txt') as openfileobject:
        for line in openfileobject:
            f = line.split()
            word = f[0]
            nums = map(np.float64,f[1:])
            wordvect = np.asarray(nums)
            wordvecdict[word] = wordvect
    self.wordvect = wordvecdict
     
  def construct_embeddings(self,vocab):
      embeddingMatrix = np.zeros((len(vocab.word_freq),200))
      wordInds = vocab.index_to_word
      for ind in wordInds:
          word = wordInds[ind]
          if word in self.wordvect:
           vect = self.wordvect[word]
          else:
            vect = np.random.randn(200,)
          embeddingMatrix[ind] = vect
      return embeddingMatrix
            
    