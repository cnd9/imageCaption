# -*- coding: utf-8 -*-
"""
Created on Sun May 15 04:27:51 2016

@author: ubuntu
"""
import tensorflow as tf
import numpy as np
class LSTMGenCell():
    
  def __init__(self, config,gloveObj):

    
    self.config = config
    self.nwords = gloveObj.embeddings.shape[0]
    self.add_placeholders()

    self.add_placeholders()
    self.inputVect = self.add_embedding(gloveObj)
    self.inputList = self.split_embedding(self.inputVect)
    self.rnn_outputs = self.add_model(self.inputList)
    self.outputs = self.add_projection(self.rnn_outputs)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
      
  def add_embedding(self,gloveObj):
      embeddings = tf.get_variable("Embedding",(self.nwords,self.config.embed_size))
  #    window = tf.nn.embedding_lookup(embeddings,tf.transpose(self.input_placeholder))
  #    self.embeddings = embeddings
      window = tf.nn.embedding_lookup(embeddings,self.input_placeholder)
        
      return window

  def split_embedding(self,inputVect):
      split = tf.split(1,self.config.num_steps,inputVect)
      squeeze = tf.squeeze(tf.concat(0,split),squeeze_dims=[1])
      inputs = tf.split(0,self.config.num_steps,squeeze) 
      return inputs
  def add_placeholders(self):

    self.input_placeholder = tf.placeholder(tf.int32,shape=(None,self.config.num_steps))
    self.labels_placeholder = tf.placeholder(tf.int32,shape=(None,self.config.num_steps+1))
    self.dropout_placeholder = tf.placeholder(tf.float32,shape=())
    self.weights_placeholder = tf.placeholder(tf.float32,shape=(None,))
    self.image_placeholder = tf.placeholder(tf.float32,shape=(None,self.config.image_dim*self.config.image_dim,self.config.image_depth))
  #  self.image_init_placeholder = tf.placeholder(tf.float32,shape=(None,self.config.image_size))
       
  def add_model(self, inputs):

    with tf.variable_scope("RNNLSTM"):
        Wf = tf.get_variable("Wf",(self.config.image_projection+self.config.embed_size+self.config.hidden_size,self.config.cell_size))
        Wi = tf.get_variable("Wiw",(self.config.image_projection+self.config.embed_size+self.config.hidden_size,self.config.cell_size))
        Wc = tf.get_variable("Wcw",(self.config.image_projection+self.config.embed_size+self.config.hidden_size,self.config.cell_size))
        Wo = tf.get_variable("Wo",(self.config.image_projection+self.config.embed_size+self.config.hidden_size,self.config.cell_size))
        Wp = tf.get_variable("Wp",(self.config.hidden_size,self.config.image_depth))        
        Wproj2=tf.get_variable("Wproj2",(self.config.image_depth,self.config.image_projection))
        bproj2=tf.get_variable("bproj2",(self.config.image_projection,),initializer=tf.constant_initializer(0.0))
  #      Wci = tf.get_variable("Wci",(2*self.config.embed_size,self.config.cell_size))
        bf = tf.get_variable("bf",(self.config.cell_size,),initializer=tf.constant_initializer(0.0))
        bi = tf.get_variable("bi",(self.config.cell_size,),initializer=tf.constant_initializer(0.0))
        bc = tf.get_variable("bc",(self.config.cell_size,),initializer=tf.constant_initializer(0.0))
        bo = tf.get_variable("bo",(self.config.cell_size,),initializer=tf.constant_initializer(0.0))
        bp = tf.get_variable("bp",(self.config.image_depth,),initializer=tf.constant_initializer(0.0)) 
  # self.weight=Wf
#    tf.add_to_collection("TotalLoss",self.config.l2*(tf.nn.l2_loss(Wf)+tf.nn.l2_loss(Wi)+tf.nn.l2_loss(Wc)+tf.nn.l2_loss(Wo)+tf.nn.l2_loss(Wp))
#)
   # tf.add_to_collection("TotalLoss",self.config.l2*(tf.nn.l2_loss))

    self.hstate = tf.zeros((self.config.batch_size,self.config.hidden_size))
    self.cstate = tf.zeros((self.config.batch_size,self.config.hidden_size))
    self.zin = tf.zeros((self.config.batch_size,self.config.image_projection))  #this should be batchsize*imagesize

    hinit = self.hstate#self.initial_hidden
    cinit = self.cstate#self.initial_cell
    zin = self.zin   
    rnn_outputs = []

    for i in xrange(len(inputs)):
        
        ##standard computing 
        f = tf.sigmoid(tf.matmul(tf.concat(1,[inputs[i],hinit,zin]),Wf)+bf)
        inp = tf.sigmoid(tf.matmul(tf.concat(1,[inputs[i],hinit,zin]),Wi)+bi)
        cell = tf.tanh(tf.matmul(tf.concat(1,[inputs[i],hinit,zin]),Wc)+bc)
        cellFull = tf.mul(f,cinit)+tf.mul(inp,cell)
        o = tf.sigmoid(tf.matmul(tf.concat(1,[inputs[i],hinit,zin]),Wo)+bo)
        hfinal = tf.mul(o,tf.tanh(cellFull))
        
        zmask = (tf.matmul(hfinal,Wp)+bp)  #batch_size x imagesie
        #resh1 = tf.reshape(self.image_placeholder,(-1,self.config.image_depth))      #from nonex3000x256 to   
        #resh2 = tf.reshape(zmask,(self.config.image_depth,1))
        #candidates = tf.matmul(resh1,zmask)
        zspl = tf.split(0,self.config.batch_size,zmask)
        respl = tf.split(0,self.config.batch_size,self.image_placeholder)  ##list of 3000ish x 256
        zout = []#tf.zeros((self.config.batch_size,self.config.image_depth))
        for j in xrange(self.config.batch_size):
          im =tf.reshape(respl[j],(-1,self.config.image_depth))
          scales = tf.matmul(im/60,tf.reshape(zspl[j],(self.config.image_depth,1)))  #3000x1  might have matrix dims wrong         
     #     self.scalestest=scales
          scales = tf.nn.softmax(tf.reshape(scales,(1,-1))) #3000x1
          z = tf.matmul(scales,im/60) #(1x3000) x (3000x256) = 1*256
          zout.append(z)
        zin = tf.matmul(tf.concat(0,zout),Wproj2)+bproj2
        cinit = cellFull
        hinit = hfinal
        hdrp=tf.nn.dropout(hfinal,self.dropout_placeholder) 
  #     zin = zout        
        rnn_outputs.append(hdrp)
    self.final_hidden = hfinal
    self.final_cell = cellFull
    self.final_z = zin
    self.scale=scales
    self.zmask=zmask
#    self.wptest = Wp
#    self.bptest = bp
#    self.bftest = bp
#    self.witest = Wi
#    self.wftest = Wf
#   # self.wo
#    self.zmask = zmask
#    self.otest=o
#    self.itest=inp
    return rnn_outputs  

 

  def add_projection(self, rnn_outputs):
  
    U = tf.get_variable("U",(self.config.hidden_size,self.nwords))
    b2 = tf.get_variable("b2",(self.nwords,),initializer=tf.constant_initializer(0.0))
    allwords = tf.concat(0,rnn_outputs)
    outputs = tf.split(0,self.config.num_steps,(tf.matmul(allwords,U)+b2))
    ### END YOUR CODE
    return outputs
