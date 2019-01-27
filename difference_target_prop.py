#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:48:49 2018

@author: will
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mnist import MNIST

def softmax(x):
   norm = 0
   mx = max(x)
   for i in range(len(x)):
      norm += np.exp(x[i]-mx)
   return np.exp(x-mx)/norm


def deriv_softmax(s):
   return s*(1-s)



def base_non_linearity(x):
    if (x>500):
      return x
    return np.log(1+np.exp(x))

non_linearity = np.vectorize(base_non_linearity)

def deriv(x):
   return 1/(1+np.exp(-x))

def gen_one_hot(x):
   out = np.zeros(10)
   out[x] = 1
   return out

def calc_accuracy(num):
   tot_correct = 0
   for i in tqdm(range(num)):
         hidden_layer = non_linearity(input_to_hidden_weights.dot(test_images[i])).reshape((hidden_layer_size,1))
         output_layer = softmax(hidden_to_output_weights.dot(hidden_layer))
         pred        = np.argmax(output_layer)
         if(pred == test_labels[i]):
            tot_correct += 1
   return tot_correct/num
      
      
   
mndata = MNIST('/home/will/Dropbox/isicni_project/MNIST_data')
mndata.gz = True
images, labels = mndata.load_training()

test_images, test_labels = mndata.load_testing()


images = np.array(images)
labels = np.array(labels)
test_images = np.array(test_images)
test_labels  = np.array(test_labels)

images = images/255
test_images = test_images/255

one_hot_labels = []

for i in range(len(labels)):
   one_hot_labels.append(gen_one_hot(labels[i]))
one_hot_labels = np.array(one_hot_labels)


learning_rate = 0.01
auto_learning_rate = 0.01

input_size = 784
hidden_layer_size = 50
output_size = 10

bias_input_to_hidden = np.random.uniform(-1,1,(hidden_layer_size,1))
input_to_hidden_weights = np.random.uniform(-1,1,(hidden_layer_size,input_size))
hidden_to_output_weights = np.random.uniform(-1,1,(output_size,hidden_layer_size))
bias_hidden_to_output = np.random.uniform(-1,1,(output_size,1))

autoencoder_output_to_hidden = np.random.uniform(-1,1,(hidden_layer_size,output_size))
bias_autoencoder = np.random.uniform(-1,1,(hidden_layer_size,1))

num_batches = 1

inv_dist_store = []

#acc = calc_accuracy(len(test_images))
#print(acc)

for _ in range(num_batches):
   
   for i in tqdm(range(len(images))):
      idx              = np.random.randint(0,len(images))
      target           = one_hot_labels[idx].reshape((output_size,1))
      hidden_layer     = non_linearity(input_to_hidden_weights.dot(images[idx]).reshape((hidden_layer_size,1))+ bias_input_to_hidden).reshape((hidden_layer_size,1))
      output_layer     = softmax(hidden_to_output_weights.dot(hidden_layer) + bias_hidden_to_output)
      db_hidden_out    = (output_layer - target).reshape((output_size,1))
      dw_hidden_output = np.dot((output_layer - target),hidden_layer.T)
      
      hidden_to_output_weights += -learning_rate*dw_hidden_output -learning_rate*0.1*hidden_to_output_weights 
      bias_hidden_to_output    += -learning_rate*db_hidden_out -learning_rate*0.1*bias_hidden_to_output 
      
      
      noisy_hidden = (hidden_layer + np.random.normal(0,0.0001,(hidden_layer_size,1))).reshape((hidden_layer_size,1))
      noisy_output = softmax(hidden_to_output_weights.dot(noisy_hidden) + bias_hidden_to_output )
      
      auto_hidden  = non_linearity(autoencoder_output_to_hidden.dot(noisy_output)+ bias_autoencoder ).reshape((hidden_layer_size,1))
      
      auto_deriv = deriv(autoencoder_output_to_hidden.dot(noisy_output).reshape((hidden_layer_size,1)) +bias_autoencoder ).reshape((hidden_layer_size,1))
      if(i%5000):
         inv_dist_store.append(np.mean(np.abs(non_linearity(autoencoder_output_to_hidden.dot(output_layer)).reshape((hidden_layer_size,1)) - hidden_layer)))
      
      dw_auto           = np.dot((auto_hidden - noisy_hidden)*auto_deriv,noisy_output.T) 
      dw_auto_bias      = (auto_hidden - noisy_hidden)*auto_deriv
      
      autoencoder_output_to_hidden  += -auto_learning_rate*dw_auto - auto_learning_rate*0.1*autoencoder_output_to_hidden
      bias_autoencoder              += -auto_learning_rate*dw_auto_bias  - auto_learning_rate*0.1*bias_autoencoder   
      
      target_hidden = hidden_layer + non_linearity(autoencoder_output_to_hidden.dot(target)+bias_autoencoder ) - non_linearity(autoencoder_output_to_hidden.dot(output_layer)+bias_autoencoder )
      
      der                      = deriv(input_to_hidden_weights.dot(images[idx]).reshape((hidden_layer_size,1)) +bias_input_to_hidden).reshape((hidden_layer_size,1))
      
      dw_input_hidden     = np.dot((hidden_layer - target_hidden)*der,(images[idx].reshape(784,1)).T)
      db_input_hidden     = (hidden_layer - target_hidden)*der
      
      input_to_hidden_weights  += -learning_rate*dw_input_hidden - learning_rate*0.05*input_to_hidden_weights
      bias_input_to_hidden     += -learning_rate*db_input_hidden - learning_rate*0.1*bias_input_to_hidden
   acc = calc_accuracy(len(test_images))
   print(acc)
   auto_learning_rate = auto_learning_rate/1.05
   learning_rate     = learning_rate/1.05

inv_dist_store = np.array(inv_dist_store)
