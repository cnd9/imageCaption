# -*- coding: utf-8 -*-
"""
Created on Sun May 15 21:43:50 2016

@author: ubuntu
"""
from DataLoader import COCO
import numpy as np
import scipy.misc

coco = COCO(31,64,100,50)
indlist = coco.imgs['VAL'].keys()
for j in xrange(20):#imindex in coco.imgs[keyword][14000:]:
                     
          #          if imindex == 393241 or imindex == 275339:
    imageData = (coco.imgs['VAL'][indlist[j]])
    imfile = imageData['file_name']
    print imageData['coco_url']
    imfile = '/home/ubuntu/May14/val2014/'+imfile
    print 
        #    print imfile
            #file = cStringIO.StringIO(urllib2.urlopen(imurl).read())
    img = scipy.misc.imread(imfile)
         #   img = Image.open(imfile)
        #    img = np.array(img.getdata()).astype(np.uint8).reshape((img.size[0],img.size[1],3))
      #      print img
    if len(img.shape) != 3:
        img = np.dstack((img,img,img))
         #   print img
          #  print img.shape
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]

    croppedImfile = scipy.misc.imresize(crop_img,(224,224))
    croppedImFile = (croppedImfile.astype(np.float32))/255.0
   # croppedImfile = np.reshape(croppedImFile,(1,224,224,3))
    scipy.misc.imshow(croppedImfile)