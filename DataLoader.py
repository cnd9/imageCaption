# -*- coding: utf-8 -*-
"""
Created on Sat May  7 19:40:48 2016

@author: ubuntu
"""

import json
import scipy.misc

import datetime
import time
import numpy as np
import urllib
import copy
import itertools
#import mask
import os
import random
from collections import defaultdict
import pickle
import time
import tensorflow as tf

class COCO:
    def __init__(self, maxLength,batchSize,hiddenSize,embedSize):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset#
        tic = time.time()
        self.ValDataset = json.load(open('/home/ubuntu/FinalProject/captions_val2014.json', 'r'))
        print 'Done (t=%0.2fs)'%(time.time()- tic)
        tic = time.time()
        self.TrainDataset = json.load(open('/home/ubuntu/FinalProject/captions_train2014.json', 'r'))
        print 'Done (t=%0.2fs)'%(time.time()- tic) 

        self.imgToAnns = {}
        self.imgs = {}
        self.word_to_index = {}
        self.index_to_word = {}
        
        self.sent_by_length = {}
        self.sent_by_length['TRAIN'] = {}
        self.sent_by_length['VAL'] = {}
        
        self.badIds = []
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = '<unk>'
        self.add_word(self.unknown, count=0)
        self.max_length = maxLength

        self.createIndex('Val')
        self.createIndex('Train')
        self.processAnns() 
        self.removeInfrequentWords() #there are a crazy amount of typos in MSCOCO, try to remove
        self.buildWordIndex()        
        self.total_words = float(sum(self.word_freq.values()))
        print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))  
        self.loaded_Train_Data = {}
        self.loaded_Val_Data = {}

        self.Train_data_iterator(batchSize,hiddenSize,embedSize)
        self.Val_data_iterator(batchSize,hiddenSize,embedSize)
        
    def createIndex(self,keyword):
        # create index
        if keyword is 'Val':
          print 'creating val index...'
          imgToAnns = {}
          imgs = {}
          imgToAnns = {ann['image_id']: [] for ann in self.ValDataset['annotations']}
          for ann in self.ValDataset['annotations']:
             imgToAnns[ann['image_id']] += [ann]

          self.imgToAnns['VAL'] = imgToAnns
          imgs      = {im['id']: {} for im in self.ValDataset['images']}
          for img in self.ValDataset['images']:
             imgs[img['id']] = img
          self.imgs['VAL'] = imgs
          print 'Val index created!'
        elif keyword is 'Train':
          print 'creating train index...'
          imgToAnns = {}
          imgs = {}
          imgToAnns = {ann['image_id']: [] for ann in self.TrainDataset['annotations']}
          for ann in self.TrainDataset['annotations']:
             imgToAnns[ann['image_id']] += [ann]

          self.imgToAnns['TRAIN'] = imgToAnns
          imgs = {im['id']: {} for im in self.TrainDataset['images']}
          for img in self.TrainDataset['images']:
             imgs[img['id']] = img
          self.imgs['TRAIN'] = imgs
          print 'Train index created!'
          
    def processAnns(self):
         for keyword in self.imgToAnns: #keywords are "TRAIN", "VAL"
            print keyword
            for imgKey in self.imgToAnns[keyword]:
              for annData in (self.imgToAnns[keyword])[imgKey]:
                  if annData['id'] not in self.badIds:
                      annotation = self.cleanUp((annData['caption']).lower())
                      self.add_sentence(annotation,imgKey,keyword)
                
    def cleanUp(self,annotation):
        wordlist = annotation.split()

        wordlist = annotation.split()
        ind = 0
        while ind < len(wordlist):
          word = wordlist[ind]
          if word[-1]==",":
            wordlist[ind] = word[:-1]
            wordlist.insert(ind+1,',')
            if word[0] == '"':
              wordlist[ind] = word[1:]
            if word[-1] == '.' or word [-1] == '"':
              wordlist[ind] = word[:-1]
            ind+=2
          else:

            if word[0] == '"':
              wordlist[ind] = word[1:]
            if word[-1] == '.' or word [-1] == '"':
              wordlist[ind] = word[:-1]
            ind+=1
     
        if ' ' in wordlist:
            wordlist.remove(' ')
        if '' in wordlist:
            wordlist.remove('')
        if len(wordlist[-1])>0:
          if wordlist[-1][-1] == '.':
            wordlist[-1] = wordlist[-1][0:-1]
        beg = ['<bos>']
        beg.extend(wordlist)
        wordlist = beg
        annotation = " ".join(wordlist)

        return annotation
                
    def add_sentence(self,annotation,imgkey,keyword):
        length = len(annotation.split())
        if length <= self.max_length - 3:
            appendage = ' <eos>'*(self.max_length-length)
            annotation += appendage
            if length not in self.sent_by_length[keyword]:
                (self.sent_by_length[keyword])[length] = {'Ann': [annotation], 'ImgId': [imgkey]}
            else:
                (((self.sent_by_length[keyword])[length])['Ann']).append(annotation)
                (((self.sent_by_length[keyword])[length])['ImgId']).append(imgkey)
            for word in annotation.split()[:length+1]:
                self.add_word(word)
         #   self.add_word('<eos>')


    def add_word(self, word, count=1):
        self.word_freq[word] += count    
        
    def buildWordIndex(self):
        for word in self.word_freq:
            if word not in self.word_to_index:
                index = len(self.word_to_index)
                self.word_to_index[word] = index
                self.index_to_word[index] = word
            
    def encodeList(self, sentenceList):
      nsent = len(sentenceList)
      wordArray = np.zeros((nsent,self.max_length))
      for i in xrange(nsent):

          wordArray[i]=np.array([self.encode(word) for word in (sentenceList[i].split())],dtype=np.int32)
      return wordArray
      
    def encode(self, word):
      if word not in self.word_to_index:
        word = self.unknown
      return self.word_to_index[word]

    def decode(self, index):
      return self.index_to_word[index]
      
    def removeInfrequentWords(self):
        for key,value in self.word_freq.items():
            if value < 3:
                if key != self.unknown:
                    self.word_freq.pop(key)

      
    def Train_data_iterator(self,batch_size,hidden_size,embed_size):
  # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
  #raw_data = np.array(raw_data, dtype=np.int32)
      TrainSentences = self.sent_by_length['TRAIN']
      count = 0
      for slength in TrainSentences:
          print slength
          sent = TrainSentences[slength]['Ann']
          ids = TrainSentences[slength]['ImgId']          
          encodedSent = self.encodeList(sent)
          #below, randomly shuff
          #rng_state = np.random.get_state()
          randint = np.random.randint(1,10000)
          np.random.seed(randint)
          np.random.shuffle(encodedSent)
          np.random.seed(randint)
          #np.random.set_state(rng_state)
          np.random.shuffle(ids)
          lwbatch = np.concatenate((np.zeros((1,)),np.ones((slength+2,)),np.zeros((self.max_length - 1 - slength - 2,))))
          lossweights = np.tile(lwbatch,batch_size)
          nsent = encodedSent.shape[0]
          print nsent

          nbatch = np.floor(nsent/batch_size)
          print "nbatch"
          print nbatch
          if nbatch>0:
            for i in xrange(int(nbatch)):
              x = encodedSent[batch_size*i:batch_size*(i+1),0:-1]
             # print x.shape[0]
              if x.shape[0]==16:
                  y = encodedSent[batch_size*i:batch_size*(i+1),:]
                  imids = ids[batch_size*i:batch_size*(i+1)]
                  self.loaded_Train_Data[count] = {'x': x,'y': y,'lossweights':lossweights,'imids':imids}
                  count+=1
            
    def Val_data_iterator(self,batch_size,hidden_size,embed_size):
  # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
  #raw_data = np.array(raw_data, dtype=np.int32)
      ValSentences = self.sent_by_length['VAL']
    #[slength]['Ann']
      count = 0
      for slength in ValSentences:
          print slength
          sent = ValSentences[slength]['Ann']
          ids = ValSentences[slength]['ImgId']          
          encodedSent = self.encodeList(sent)
          #print encodedSent
          #rng_state = np.random.get_state()
          randint = np.random.randint(1,10000)
          np.random.seed(randint)
          np.random.shuffle(encodedSent)
          np.random.seed(randint)
          #np.random.set_state(rng_state)
          np.random.shuffle(ids)
          lwbatch = np.concatenate((np.zeros((1,)),np.ones((slength+2,)),np.zeros((self.max_length - 1 - slength - 2,))))
          lossweights = np.tile(lwbatch,batch_size)
          nsent = encodedSent.shape[0]
          print nsent

          nbatch = np.floor(nsent/batch_size)

          if nbatch>0:
            for i in xrange(int(nbatch)):
              x = encodedSent[batch_size*i:batch_size*(i+1),0:-1]
              if x.shape[0]==16:
                  y = encodedSent[batch_size*i:batch_size*(i+1),:]
                  imids = ids[batch_size*i:batch_size*(i+1)]
                  self.loaded_Val_Data[count] = {'x': x,'y': y,'lossweights':lossweights,'imids':imids}
                  count+=1


def imageDataIterator(keyword,coco,session,shallowop,deepop,plh):
  
 if keyword == 'TRAIN':
   trainData = coco.loaded_Train_Data
   baseDir = '/home/ubuntu/May14/train2014/'
 elif keyword == 'VAL':
   trainData = coco.loaded_Val_Data
   baseDir = '/home/ubuntu/May14/val2014/'
 keylist = trainData.keys()
 np.random.shuffle(keylist)
 if keyword == 'TRAIN':
   maxrange = len(keylist)
 elif keyword == 'VAL':
   maxrange = 10000
 for count in xrange(maxrange):
#     t0 = time.time()
     x = trainData[keylist[count]]['x']
     y = trainData[keylist[count]]['y']
     lossweights = trainData[keylist[count]]['lossweights']
     imgs = trainData[keylist[count]]['imids']
     imdata = np.zeros((x.shape[0],224,224,3))
 #    t1=time.time()
     for imcount in xrange(len(imgs)):
         filename = baseDir + coco.imgs[keyword][imgs[imcount]]['file_name']
         imdata[imcount]=imagePreprocess(filename)
      #   shallow_out[imcount]=mydat[imgs[imcount]]
     feed_dict = {plh:imdata}       
 #    t2=time.time()
     #with tf.device('/gpu:0'):
     deep_out,shallow_out = session.run([deepop,shallowop], feed_dict=feed_dict)
     #print "x"
    # print x
     deep_out=np.reshape(deep_out,(x.shape[0],14*14,512))  ###make sure does what you expect
     yield(x,y,deep_out,shallow_out,lossweights)



def COCOimageDataIterator(coco,session,shallowop,deepop,plh):
 
  baseDir = '/home/ubuntu/May14/val2014/'
  imgData = coco.imgs['VAL']
  keylist = imgData.keys()
  maxrange = 10000
  
  
  for count in xrange(maxrange):
     imid=keylist[count] 
     imdata = np.zeros((1,224,224,3))

     filename = baseDir +imgData[imid]['file_name']
     imdata=imagePreprocess(filename)
     feed_dict = {plh:imdata}       

     deep_out,shallow_out = session.run([deepop,shallowop], feed_dict=feed_dict)
     deep_out=np.reshape(deep_out,(1,14*14,512))  ###make sure does what you expect
     yield(imid,deep_out,shallow_out)  
  
  

def imagePreprocess(imfile):
  #  imfile = imageData['file_name']
  #  imfile = '/home/ubuntu/May14/val2014/'+imfile
        #    print imfile
            #file = cStringIO.StringIO(urllib2.urlopen(imurl).read())
    img = scipy.misc.imread(imfile)

    if len(img.shape) != 3:
        img = np.dstack((img,img,img))

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]

    croppedImfile = scipy.misc.imresize(crop_img,(224,224))
    croppedImFile = (croppedImfile.astype(np.float32))/255.0
    croppedImfile = np.reshape(croppedImFile,(1,224,224,3))
    return croppedImfile  

