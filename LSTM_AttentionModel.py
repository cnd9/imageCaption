# -*- coding: utf-8 -*-
"""
Created on Sun May  1 14:31:34 2016

@author: cdonnelly


"""
import matplotlib
import pickle
matplotlib.use('Agg')
import sys
sys.path.append('/home/ubuntu/May14/COCOEval/coco-caption-master/')
import matplotlib.pyplot as plt
from runEval import evaluate
import time
import numpy as np
from copy import deepcopy
import random
import scipy.misc

from DataLoader import COCO, imageDataIterator, COCOimageDataIterator
from utils import sample
from wordVecUtils import WORDVECS
from LSTMGenCell import LSTMGenCell

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
import json


class Config(object):
   #Holds model hyperparams and data information.

  def __init__(self, maxsteps):

    self.batch_size = 16#64#128#64#2#64
    self.embed_size = 200
    self.hidden_size = 500
    self.cell_size = 500
    self.num_steps = maxsteps
    self.max_epochs = 16#16#100#16
    self.early_stopping = 2
    self.dropout = 0.7
    self.lr = 0.0003
    self.image_size = 4096
    self.l2 = .00005
    self.image_dim = 14
    self.image_depth = 512
    self.image_projection = 512

    
class LSTMCell():
    
  def __init__(self, config,gloveObj):
    self.config = config
    self.nwords = gloveObj.embeddings.shape[0]
    self.add_placeholders()
    
    self.inputs = self.add_embedding(gloveObj)
    self.imageInputs = self.add_image_projection()
    self.inputsTot = []
    self.inputsTot.append(self.imageInputs)
    self.inputsTot = self.inputsTot + self.inputs
    self.rnn_outputs = self.add_model(self.inputsTot)
    self.outputs = self.add_projection(self.rnn_outputs)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    output = tf.reshape(tf.concat(1, self.outputs), [-1, self.nwords])
    self.calculate_loss,self.nol2 = self.add_loss_op(output)
    self.train_step = self.add_training_op(self.calculate_loss)  

      
  def add_embedding(self,gloveObj):

          #Tensorflow word embeddings from pre-trained Glove object
    embeddings = tf.get_variable("Embedding",(self.nwords,self.config.embed_size),initializer = tf.constant_initializer(gloveObj.embeddings))
    window = tf.nn.embedding_lookup(embeddings,self.input_placeholder)
    split = tf.split(1,self.config.num_steps,window)
    squeeze = tf.squeeze(tf.concat(0,split),squeeze_dims=[1])
    inputs = tf.split(0,self.config.num_steps,squeeze)    
    self.embeddings = embeddings
    return inputs

     
  def add_placeholders(self):

    self.input_placeholder = tf.placeholder(tf.int32,shape=(None,self.config.num_steps))
    self.labels_placeholder = tf.placeholder(tf.int32,shape=(None,self.config.num_steps+1))
    self.dropout_placeholder = tf.placeholder(tf.float32,shape=())
    self.weights_placeholder = tf.placeholder(tf.float32,shape=(None,))
    self.image_placeholder = tf.placeholder(tf.float32,shape=(None,self.config.image_dim*self.config.image_dim,self.config.image_depth))
    self.image_init_placeholder = tf.placeholder(tf.float32,shape=(None,self.config.image_size))

  def add_image_projection(self):
      
      # Project VGGnet 4096-dimensional image feature vector into the 200-dimensional word embedding space
    with tf.variable_scope('RNNLSTM'):
        Wim = tf.get_variable("Wim",(self.config.image_size,self.config.embed_size))
        Bim = tf.get_variable("Bim",self.config.embed_size)
        projectedImageInput = tf.matmul(self.image_init_placeholder,Wim) + Bim
        self.wproj=Wim
        self.bproj=Bim
        return projectedImageInput

  def add_model(self, inputs):

    with tf.variable_scope("RNNLSTM"):
        Wf = tf.get_variable("Wf",(self.config.image_projection+self.config.embed_size+self.config.hidden_size,self.config.cell_size))
        Wi = tf.get_variable("Wiw",(self.config.image_projection+self.config.embed_size+self.config.hidden_size,self.config.cell_size))
        Wc = tf.get_variable("Wcw",(self.config.image_projection+self.config.embed_size+self.config.hidden_size,self.config.cell_size))
        Wo = tf.get_variable("Wo",(self.config.image_projection+self.config.embed_size+self.config.hidden_size,self.config.cell_size))
        Wp = tf.get_variable("Wp",(self.config.hidden_size,self.config.image_depth))        
        Wproj2=tf.get_variable("Wproj2",(self.config.image_depth,self.config.image_projection))
        bproj2=tf.get_variable("bproj2",(self.config.image_projection,),initializer=tf.constant_initializer(0.0))
        bf = tf.get_variable("bf",(self.config.cell_size,),initializer=tf.constant_initializer(0.0))
        bi = tf.get_variable("bi",(self.config.cell_size,),initializer=tf.constant_initializer(0.0))
        bc = tf.get_variable("bc",(self.config.cell_size,),initializer=tf.constant_initializer(0.0))
        bo = tf.get_variable("bo",(self.config.cell_size,),initializer=tf.constant_initializer(0.0))
        bp = tf.get_variable("bp",(self.config.image_depth,),initializer=tf.constant_initializer(0.0)) 
  # self.weight=Wf
    tf.add_to_collection("TotalLoss",self.config.l2*(tf.nn.l2_loss(Wproj2)+tf.nn.l2_loss(Wf)+tf.nn.l2_loss(Wi)+tf.nn.l2_loss(Wc)+tf.nn.l2_loss(Wo)+tf.nn.l2_loss(Wp))
)
   # tf.add_to_collection("TotalLoss",self.config.l2*(tf.nn.l2_loss))

    self.hstate = tf.zeros((self.config.batch_size,self.config.hidden_size))
    self.cstate = tf.zeros((self.config.batch_size,self.config.hidden_size))
    self.zin = tf.zeros((self.config.batch_size,self.config.image_projection))  #this should be batchsize*imagesize (512)

    hinit = self.hstate
    cinit = self.cstate
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
        
        zmask = (tf.matmul(hfinal,Wp)+bp)  #batch_size x imagesize

        zspl = tf.split(0,self.config.batch_size,zmask)
        respl = tf.split(0,self.config.batch_size,self.image_placeholder)  ##list of 3000ish x 256
        zout = []#tf.zeros((self.config.batch_size,self.config.image_depth))
        for j in xrange(self.config.batch_size):
          im =tf.reshape(respl[j],(-1,self.config.image_depth))
          scales = tf.matmul(im/60,tf.reshape(zspl[j],(self.config.image_depth,1)))  #3000x1   
          scales = tf.nn.softmax(tf.reshape(scales,(1,-1))) #3000x1
          z = tf.matmul(scales,im/60) #(1x3000) x (3000x256) = 1*256
          zout.append(z)
        zin = tf.matmul(tf.concat(0,zout),Wproj2)+bproj2
        cinit = cellFull
        hinit = hfinal
        hdrp=tf.nn.dropout(hfinal,self.config.dropout) 
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
    outputs = tf.split(0,self.config.num_steps+1,(tf.matmul(allwords,U)+b2))
    return outputs
    
  def add_loss_op(self, output):
    #Add loss ops to the computational graph.

    self.logits = output
    self.targets = tf.reshape(self.labels_placeholder,[-1])
    self.weights = self.weights_placeholder
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits,self.targets)
    loss = tf.reduce_sum(tf.mul(self.weights,loss))/tf.reduce_sum(self.weights)
    allLosses = tf.get_collection("TotalLoss")
    print loss
    return (loss + allLosses[0],loss)   
    
  def add_training_op(self, loss):
    opt = tf.train.AdamOptimizer(learning_rate = self.config.lr)
    dummy_total=tf.constant(0.0)
    for v in tf.all_variables():
        dummy_total+=tf.reduce_sum(v)
    self.dummy_minimizer=opt.minimize(dummy_total)
    train_op = opt.minimize(loss)

    return train_op
  
def run_epoch(session, gen_model,gen_config,shallowOp,deepOp,imPlaceholder,model, saver,checkpoint_dir,epoch,train_op=None,verbose=100):

    dp = model.config.dropout
    if not train_op:
      train_op = tf.no_op()
      keyword = 'VAL'
      dp = 1
    else:
      keyword= 'TRAIN'
    total_loss = []    

    step=0
    vocab = COCO(33,16,500,200)
    for (x,y,deep_out,shallow_out,lossweights) in imageDataIterator(keyword,vocab,session,shallowOp,deepOp,imPlaceholder):      

      feed = {model.input_placeholder: x,
              model.labels_placeholder: y,
              model.weights_placeholder : lossweights,
              model.image_placeholder : deep_out,
              model.image_init_placeholder : shallow_out,
              model.dropout_placeholder: dp}
      
      loss,_ = session.run([model.nol2,train_op], feed_dict=feed)

      total_loss.append(loss)
          
      if verbose and step % (verbose) == 0:
        sys.stdout.write('\r{} : pp = {}'.format(step, np.exp(np.mean(total_loss))))
        print "loss:"
        print loss
        sys.stdout.flush()
      if step%1000==0:
        if keyword == 'TRAIN':
          saver.save(session,checkpoint_dir+'model.ckpt',global_step=epoch+1)
          scales = session.run(model.scale,feed_dict=feed)

          print "max scale"
          print np.max(scales)
          print "saved weights"
 
      if step%10000==0:
        resultslist = []
        pmatrix,pbias = session.run([model.wproj,model.bproj])
        count=0
        for (imid,deepout,shallowout) in COCOimageDataIterator(vocab,session,shallowOp,deepOp,imPlaceholder):
          img = np.dot(shallowout,pmatrix)+pbias
          ann= ' '.join(generate_text(count,
                 session, gen_model, vocab,gen_config, starting_text='<bos>', image=img,convim=deepout,temp=.05)[1:-1])
          dic={}
          dic={"image_id":imid,"caption":ann}
          resultslist.append(dic)
          if count%500==0:
            print ann
            print ann.split()
          count+=1
        with open('/home/ubuntu/May14/COCOEval/coco-caption-master/results/captions_val2014_realcapMay27_results.json', 'w') as outfile:
          json.dump(resultslist, outfile)
        evaluate()

      step+=1
    if verbose:
      sys.stdout.write('\r')
    return np.exp(np.mean(total_loss)) 
    
def test_LSTM():

  config = Config(32)  #40=maxsteps
  vocab = COCO(33,16,500,200) #41 = maxsteps+1, 64=batchsize, 150=hidden size
 # loaded_Train_Data = pickle.load(open('/home/ubuntu/May14/SavedTrainData','rb'))

  gloveObj = WORDVECS(vocab)
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1
  starting_text = '<bos>'


  with open("vgg16-20160129.tfmodel", mode='rb') as f:
    fileContent = f.read()
  print "done reading"
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(fileContent)
  imPlaceholder = tf.placeholder("float", [None, 224, 224, 3])
  tf.import_graph_def(graph_def, input_map={ "images": imPlaceholder })
  print "graph loaded from disk"
  graph = tf.get_default_graph()
  shallowOp = graph.get_tensor_by_name("import/Relu_1:0")
  deepOp = graph.get_tensor_by_name("import/conv5_3/Relu:0")

  trainpp = []
  valpp = []
  
  with tf.variable_scope('RNNLSTMgen') as scope:
    model = LSTMCell(config,gloveObj)
    scope.reuse_variables()
    gen_model = LSTMGenCell(gen_config,gloveObj)

  
  init = tf.initialize_all_variables()
  checkpoint_dir = r'/home/ubuntu/May14/May27BiggerProjPart2/'
 # checkpoint_steps=2
  with tf.Session() as session:
    session.run(init)  ##what does that do?
    var= tf.trainable_variables()
    for i in xrange(len(var)):
      print (var[i])
    var= tf.all_variables()
    for i in xrange(len(var)):
      print (var[i].name)
    saver = tf.train.Saver(var_list=var)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(session,ckpt.model_checkpoint_path)
      print "restored weights"
      print ckpt.model_checkpoint_path

    for epoch in xrange(config.max_epochs):
      print "try to save"
   #   saver.save(session,checkpoint_dir+'model.ckpt',global_step=epoch+1)
      print 'Epoch {}'.format(epoch)
#
      train_pp = run_epoch(
          session, gen_model,gen_config,shallowOp,deepOp,imPlaceholder,model,
          saver,checkpoint_dir,epoch,train_op=model.train_step)
      print 'Training perplexity: {}'.format(train_pp)
      saver.save(session,checkpoint_dir+'model.ckpt',global_step=epoch+1)
      print "saved weights"
      trainpp.append(train_pp)

      pickle.dump(trainpp,open(checkpoint_dir+"trainLoss.p",'wb'))


#### Uncomment for testing caption generation
#      if epoch%2 == 0:#config.max_epochs:
#        valid_pp = run_epoch(session, shallowOp,deepOp,imPlaceholder,model,
#          saver,checkpoint_dir,epoch)
#        print 'Validation perplexity: {}'.format(valid_pp)
#      valpp.append(valid_pp)
#      pickle.dump(valpp,open(checkpoint_dir+"valLoss.p",'wb'))
#      if epoch%2 == 0:      
#        resultslist = []
#        pmatrix,pbias = session.run([model.wproj,model.bproj])
#        count=0
##        for (imid,deepout,shallowout) in COCOimageDataIterator(vocab,session,shallowOp,deepOp,imPlaceholder):
#          img = np.dot(shallowout,pmatrix)+pbias
#          ann= ' '.join(generate_text(
#                 session, gen_model, gen_config, starting_text=starting_text, image=img,convim=deepout,temp=.05)[1:-1])
#          dic={}
#          dic={"image_id":imid,"caption":ann}
#          resultslist.append(dic)
#          if count%500==0:
#            print ann
#            print ann.split()
#          count+=1
#        with open('/home/ubuntu/May14/COCOEval/coco-caption-master/results/captions_val2014_realcapMay24_results.json', 'w') as outfile:
#          json.dump(resultslist, outfile)
#
#      starting_text = '<bos>'
#      shallowArray = pickle.load(open("20ValTestImages_TopLayer_May21.p","r"))  
#      pmatrix,pbias = session.run([model.wproj,model.bproj])  
#      deepList=pickle.load(open("20ValTestImages_ConvLayer_May21.p","r"))
#      img = np.dot(shallowArray,pmatrix)+pbias
#
#      for q in xrange(2):
#        for m in xrange(20):
#          print m
#          print ' '.join(generate_text(
#               session, gen_model, vocab,gen_config, starting_text=starting_text, image=img[m],convim=deepList[m],temp=.05))
#      for q in xrange(2):
#        for m in xrange(20):
#          print m
#          print ' '.join(generate_text(
#               session, gen_model, vocab,gen_config, starting_text=starting_text, image=img[m],convim=deepList[m],temp=.2))
##
#      #starting_text = raw_input('> ')


def generate_text(step,session, model, vocab,config, starting_text, image, convim,temp):

  # Imagine tokens as a batch size of one, length of len(tokens[0])
  stop_length=100
  stop_tokens = ['<eos>']
  hstate = model.hstate.eval()
  cstate = model.cstate.eval()

  tokens = [vocab.encode(word) for word in starting_text.split()]

  scales=[]
  zmasks=[]

  for i in xrange(stop_length):
    image=np.reshape(image,(1,1,200))
    if i == 0:
      feed = {model.inputVect: image,
            model.hstate: hstate,
            model.cstate: cstate,
            model.image_placeholder: convim,
            model.dropout_placeholder: 1}
      hstate, cstate, zstate, y_pred,scale,zmask = session.run([model.final_hidden, model.final_cell, model.final_z,model.predictions[-1],model.scale,model.zmask], feed_dict=feed)

    else:
        inputFeed = np.reshape(tokens[-1],(1,1))
        feed = {model.input_placeholder: inputFeed,
            model.hstate: hstate,
            model.cstate: cstate,
            model.zin: zstate,
            model.image_placeholder: convim,
            model.dropout_placeholder: 1} 
        hstate, cstate, zstate, y_pred,scale,zmask = session.run([model.final_hidden, model.final_cell, model.final_z,model.predictions[-1],model.scale,model.zmask], feed_dict=feed)
    if step==3:
        scales.append(scale)
        zmasks.append(zmask)

    if i!=0:
      next_word_idx = sample(y_pred[0], temperature=temp)
      tokens.append(next_word_idx)
    if stop_tokens and vocab.decode(tokens[-1]) in stop_tokens:
      if step==3:
            pickle.dump(scales,open("scalesSave.p","wb"))
            pickle.dump(zmasks,open("zsave.p","wb"))
      break
  output = [vocab.decode(word_idx) for word_idx in tokens]
 # print weightcheck
  return output

if __name__ == "__main__":
    test_LSTM()
