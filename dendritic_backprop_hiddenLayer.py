#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:11:18 2018

@author: will
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
from dendritic_backprop_classes import *


timestep = 1*1e-1

mult = 3

low_pass_filter_timescale_weights = 30
low_pass_filter_timescale_inputs = 3

# Learning rates
"""lr_input_pyr = 0*0.0001
lr_pyr_int  = 0.00004375*mult
lr_int_pyr = 0.0001*mult
lr_above_pyr = 0.00001
lr_in_hidden = 0.00005*mult
lr_hidden_out = 0.000005*mult"""

lr_input_pyr = 0*0.0001
lr_pyr_int  = 0.00002375*mult
lr_int_pyr = 0.0001*mult
lr_above_pyr = 0.00001
lr_in_hidden = 0.00011875
lr_hidden_out = 0.00001

global_sigma = 0



auto_set_optimal_weights = True
train_interneuron_weights = False
train_inter_pyrimidal_weights = False
train_backward_weights = False
intn_current= False


target_matrix_1 = np.random.uniform(-1,1,(20,30))
target_matrix_2 = np.random.uniform(-1,1,(10,20))

      
hidden_1 = hidden_neuron_layer(30,20,10,10,0.01)
output  = output_layer(20,10,0.01)

if(auto_set_optimal_weights):
   hidden_1.int_to_pyrA = -np.copy(hidden_1.above_to_pyrA)
   hidden_1.pyr_to_int = np.copy(output.input_to_pyrB)


vA_store = []
weight_store = []
weight_store2 = []

uI_store = []
uP_store = []
soma_store = []
input_store =[]
error_store = []
output_store = []
target_store =[]
hidden_uP_store    = []
weight_store_int_pyr  = []
weight_store_pyr_int  = []
weight_store_above_pyr = []

input_layer = np.zeros(30)


def MSE(x,y):
    y = y.reshape(x.shape)
    return np.mean((x-y)**2)

        
"""x = np.random.uniform(-1,1,30)
targets = target_matrix_2.dot(target_matrix_1.dot(x))
x = non_linearity(x)"""  

n_training_set = 500
x_list = np.random.uniform(-1,1,(n_training_set,30))
 
      
store = False
iters  = 5*5000*12
for i in tqdm(range(iters)):
   
   
   if(i%1000==0):
      if(store):
         pass
         #error_store.append(MSE(non_linearity(output.uP),non_linearity(targets)))
      r = np.random.randint(0,n_training_set)
      x = x_list[r]
      targets = target_matrix_2.dot(non_linearity(target_matrix_1.dot(non_linearity(x))))
      x = non_linearity(x)  
      store = True 

   if(i> 0.025*iters):
      train_interneuron_weights = True
      train_inter_pyrimidal_weights = False
      train_backward_weights = False
      intn_current = True

   input_layer  = low_pass_filter(input_layer,x,low_pass_filter_timescale_inputs,timestep)
   

   hidden_1.update_membrane_potentials(input_layer,output.uP,intn_current)

   
   if(i < iters):
      output.update_membrane_potentials(hidden_1.uP)
      if(train_interneuron_weights):
         hidden_1.update_pyr_to_int_weights(lr_pyr_int)
         hidden_1.update_int_to_pyr_weights(lr_int_pyr)
      if(train_backward_weights):
         hidden_1.update_above_to_pyrA_weights(output.uP,lr_above_pyr)
         
      if(train_inter_pyrimidal_weights):
         hidden_1.update_input_pyr_weights(input_layer,lr_in_hidden)
         output.update_input_pyr_weights(hidden_1.uP,lr_hidden_out)
   else:
       output.update_membrane_potentials(hidden_1.uP)
       if(train_interneuron_weights):
          pass
           # hidden_1.update_pyr_to_int_weights(lr_pyr_int)
            #hidden_1.update_int_to_pyr_weights(lr_int_pyr)
   if(i%1== 0):
      weight_store.append(np.copy(hidden_1.input_to_pyrB))
      weight_store2.append(np.copy(output.input_to_pyrB))
      uP_store.append(np.copy(output.uP))
      uI_store.append(np.copy(hidden_1.uI))
      vA_store.append(np.copy(hidden_1.vA))
      hidden_uP_store.append(np.copy(hidden_1.uP))
      output_store.append(non_linearity(np.copy(output.uP)))
      target_store.append(non_linearity(np.copy(targets)))
      input_store.append(np.copy(targets))
      error_store.append(MSE(non_linearity(output.uP),non_linearity(targets)))
      weight_store_int_pyr.append(np.copy(hidden_1.int_to_pyrA))
      weight_store_pyr_int.append(np.copy(hidden_1.pyr_to_int))
      weight_store_above_pyr.append(np.copy(hidden_1.above_to_pyrA))


vA_store = np.array(vA_store)
error_store = np.array(error_store)
weight_store= np.array(weight_store)
weight_store2 = np.array(weight_store2)
output_store = np.array(output_store)
input_store = np.array(input_store)
hidden_uP_store = np.array(hidden_uP_store)
weight_store_int_pyr  = np.array(weight_store_int_pyr) 
weight_store_pyr_int  = np.array(weight_store_pyr_int)
weight_store_above_pyr = np.array(weight_store_above_pyr)

feed_forward_input_hidden = np.copy(hidden_1.input_to_pyrB)
feed_forward_hidden_output = np.copy(output.input_to_pyrB)
feed_forward_test = []
correct_test  = []
for  i in range(200):
   if(i<100):
      x = x_list[0]
   else:
      x = x_list[1]
   feed_forward_test.append(feed_forward_hidden_output.dot(non_linearity(feed_forward_input_hidden.dot(non_linearity(x)))))
   correct_test.append(target_matrix_2.dot(target_matrix_1.dot(non_linearity(x))))

feed_forward_test = np.array(feed_forward_test)
coorect_test = np.array(correct_test)

target_store = np.array(target_store)
uP_store = np.array(uP_store)
uI_store = np.array(uI_store)
plt.plot(vA_store[:,:,0])
plt.title("Potential of Apical Dendrites")
plt.xlabel("Iterations")
plt.show()
plt.figure()
plt.plot(np.mean(np.abs(uI_store[:,:,0]- uP_store[:,:,0]),axis=1))
plt.title("Absolute difference between pyrimidal and interneuron potential")
plt.xlabel("Iterations")
plt.yscale('log')
plt.show()

