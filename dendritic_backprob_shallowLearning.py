#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:04:42 2018

@author: will
"""

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


low_pass_filter_timescale_weights = 5
low_pass_filter_timescale_inputs = 5

# Learning rates


lr_input_pyr = 0.04


global_sigma = 0.1

def MSE(x,y):
   ans = 0
   for i in range(len(x)):
      ans += (x[i] - y[i])**2
   return ans


target_matrix_2 = np.random.uniform(-1,1,(10,20))


output  = output_layer(20,10,sigma=0.01)

#test3 = hidden_neuron_layer(20,10,10,10)

vA_store = []
weight_store_int_pyr = []
weight_store_above_pyrA = []
uI_store = []
uP_store = []
soma_store = []
input_layer = np.zeros(20)

error_store = []
uP_store =[]
weight_store = []
target_store = []

n_training_set = 3


x_list = np.random.uniform(-1,1,(n_training_set,20))


store = False
iters  = int(1*1e5)
for i in tqdm(range(iters)):
   
   if(i%1000== 0):
      if(store):
         error_store.append(MSE(non_linearity(output.uP),non_linearity(targets)))
      r = np.random.randint(0,n_training_set)
      x = x_list[r]
      targets = target_matrix_2.dot(non_linearity(x)).reshape(10,1)
      x= non_linearity(x) 
      """ x = np.random.uniform(-1,1,20)
      targets = target_matrix_2.dot(x).reshape((10,1))
      x = non_linearity(x) """ 
      store =True

#   if(i%2000000==0):
 #    lr_input_pyr = lr_input_pyr/2

   input_layer  = low_pass_filter(input_layer,x,low_pass_filter_timescale_inputs,timestep)
   if(i<0.075*1e5):
            output.update_membrane_potentials(input_layer)
            output.update_input_pyr_weights(input_layer,lr_input_pyr)
   elif(i<0.8*1e5):
      output.update_membrane_potentials(input_layer,targets)
      output.update_input_pyr_weights(input_layer,lr_input_pyr)
   else:
      output.update_membrane_potentials(input_layer)
      #output.update_input_pyr_weights(input_layer,lr_input_pyr)
   if(i%1==0):
      uP_store.append(np.copy(output.uP))
      target_store.append(np.copy(targets))
 
   
error_store = np.array(error_store)
plt.plot(error_store,'.')
uP_store = np.array(uP_store)
weight_store = np.array(weight_store)
target_store = np.array(target_store)



