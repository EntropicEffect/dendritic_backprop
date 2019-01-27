#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:11:18 2018

@author: will
"""
import matplotlib.pyplot as plt
from tqdm import tqdm
from dendritic_backprop_classes import *


auto_set_optimal_weights = True
train_interneuron_weights = True
train_inter_pyrimidal_weights = True
train_backward_weights = False


      
hidden_1 = hidden_neuron_layer(30,20,10,10)
output  = output_layer(20,10)
if(auto_set_optimal_weights):
   hidden_1.int_to_pyrA = -hidden_1.above_to_pyrA
   hidden_1.pyr_to_int = output.input_to_pyrB
#test3 = hidden_neuron_layer(20,10,10,10)

vA_store = []
weight_store_int_pyr = []
weight_store_above_pyrA = []
uI_store = []
uP_store = []
soma_store = []


iters  = 10000
for i in tqdm(range(iters)):
   input_layer = np.zeros(30)
   if(i%1000== 0):
      x = np.random.uniform(-1,1,30)
      targets = target_matrix_2.dot(target_matrix_1.dot(x))
      x = non_linearity(x)   
   input_layer  = low_pass_filter(input_layer,x,low_pass_filter_timescale_inputs,timestep)
   y = np.ones(10)
   hidden_1.update_membrane_potentials(x,output.uP,1)
   output.update_membrane_potentials(hidden_1.uP,targets)

   if(train_interneuron_weights):
      hidden_1.update_pyr_to_int_weights()
      hidden_1.update_int_to_pyr_weights()
      
   if(train_inter_pyrimidal_weights):
      hidden_1.update_input_pyr_weights(input_layer)
      output.update_input_pyr_weights(hidden_1.uP)

   if(i%1== 0):
      weight_store_int_pyr.append(np.copy(hidden_1.int_to_pyrA))
      weight_store_above_pyrA .append(np.copy(hidden_1.above_to_pyrA))
      uP_store.append(np.copy(output.uP))
      uI_store.append(np.copy(hidden_1.uI))
      vA_store.append(np.copy(hidden_1.vA))


vA_store = np.array(vA_store)
weight_store_int_pyr= np.array(weight_store_int_pyr)
weight_store_above_pyrA= np.array(weight_store_above_pyrA)
uP_store = np.array(uP_store)
uI_store = np.array(uI_store)
plt.plot(vA_store[:,:,0])
plt.title("Potential of Apical Dendrites")
plt.show()
plt.figure()
plt.plot(np.mean(np.abs(uI_store[:,:,0]- uP_store[:,:,0]),axis=1))
plt.title("Mean abs value of difference between pyr soma rate and interneuron rate")
plt.yscale('log')
plt.show()

diff_vector = []
for i in range(weight_store_int_pyr.shape[0]):
   diff_vector.append(tot_weight_sum(weight_store_int_pyr[i,:,:],weight_store_above_pyrA[i,:,:]))
diff_vector = np.array(diff_vector)
plt.figure()
plt.plot(diff_vector)
