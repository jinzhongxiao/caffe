#!/usr/bin/env python    
# encoding: utf-8    
  
import cv2    
import numpy as np    
    
img = cv2.imread("/home/roger/WorkSpace/caffe/have-fun-with-machine-learning/data/untrained-samples/1.jpg")    
    

cv2.imshow("Merged", img)    
  
cv2.waitKey(0)    
cv2.destroyAllWindows()  


