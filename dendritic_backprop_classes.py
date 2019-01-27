#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:56:03 2018

@author: will
"""

import numpy as np


timestep = 1*1e-1

low_pass_filter_timescale_weights = 30
low_pass_filter_timescale_inputs = 3

# Learning rates
use_abs = False
use_linear_difference = False

lr_above_pyr = 0.0005




def tot_weight_sum(x,y):
   return np.mean(np.abs(x+y))

def base_voltage_clamp(x,v_top=150,v_low=-150):
   if(x>v_top):
      return v_top
   elif(x<v_low):
      return v_low
   else:
      return x
   

def base_non_linearity(x):
    if (x>500):
      return x
    return np.log(1+np.exp(x))

non_linearity = np.vectorize(base_non_linearity)
voltage_clamp = np.vectorize(base_voltage_clamp)

def low_pass_filter(x,target,tw,dt):
   return x + (dt/tw)*(target - x)

def elem_mult(x,y):
   if (x.shape != y.shape):
      print("xshape =",x.shape)
      print("yshapevv=",y.shape)
      raise ValueError('Vectors of unequal length')
   out = x*y
   return out
      


class neuron():
   def __init__(self,sigma):
      self.glk = 0.1
      self.gB  = 1.0
      self.gA  = 0.8
      self.gD  = 1
      self.sigma = sigma
      
      self.g_som = 0.8
      
      self.E_inh = -1
      self.E_ex  = 1
      


class hidden_neuron_layer(neuron):
   
   def __init__(self,N_in,N_pyr,N_int,N_above,sigma=0.1):
      
      super(hidden_neuron_layer,self).__init__(sigma)
      self.input_to_pyrB = 0.1*np.random.uniform(-1,1,(N_pyr,N_in))
      self.pyr_to_int = np.random.uniform(0,1,(N_int,N_pyr))
      self.int_to_pyrA = np.random.uniform(-1,0,(N_pyr,N_int))
      self.above_to_pyrA = np.random.uniform(0,1,(N_pyr,N_above))


      self.scale_factor = (self.gB + self.glk)/(self.gB+self.gA+self.glk)
      self.input_to_pyrB_delta = 0
      self.pyr_to_int_delta = 0
      self.int_to_pyrA_delta = 0
      self.above_to_pyrA_delta = 0
      
      self.N_pyr = N_pyr
      self.N_int = N_int
      self.N_above = N_above
      self.N_below = N_in
   

      self.vB = 1
      self.vA = 1
      self.vI = 1
      
      self.vA_rest = 0*np.ones((N_pyr,1))
      
      self.uI  = np.random.uniform(-1,1,(N_int,1))
      self.uP  = np.random.uniform(-1,1,(N_pyr,1))
      


   def update_membrane_potentials(self,u_below,u_above,interneuron_training=True):
      

      u_above = u_above.reshape((self.N_above,1))
      u_below = u_below.reshape((self.N_below,1))
      
      noiseP = self.sigma*np.random.normal(0,1,(self.N_pyr,1))
      noiseI = self.sigma*np.random.normal(0,1,(self.N_int,1))
      
      self.vB = voltage_clamp(self.input_to_pyrB.dot(non_linearity(u_below)).reshape((self.N_pyr,1)))
      self.vA = voltage_clamp((self.above_to_pyrA.dot(non_linearity(u_above)) + self.int_to_pyrA.dot(self.uI)).reshape((self.N_pyr,1)))
      self.vI = voltage_clamp(self.pyr_to_int.dot(non_linearity(self.uP)).reshape((self.N_int,1)))
      
      
      g_ex = self.g_som*(u_above - self.E_inh)/(self.E_ex-self.E_inh)
      g_inh = -self.g_som*(u_above- self.E_ex)/(self.E_ex - self.E_inh)
      
      if(interneuron_training):
         I_I = g_ex*(self.E_ex-self.uI) + g_inh*(self.E_inh - self.uI)
      else:
         I_I = 0
      
      self.uP += timestep*(-self.glk*self.uP + self.gB*(self.vB - self.uP) + self.gA*(self.vA - self.uP) + noiseP)
      self.uI += timestep*(-self.glk*self.uI + self.gD*(self.vI - self.uI) + I_I + noiseI) 
      
   def update_input_pyr_weights(self,u_below,lr_input_pyr):
      
      soma_rate  = non_linearity(self.uP)
      vB_hat     = self.gB/(self.glk+self.gB+self.gA)*self.vB
      basal_rate = non_linearity(vB_hat)
      
      if(use_abs):
         input_rate = np.abs(u_below).reshape((1,u_below.size))
      else:
         input_rate = non_linearity(u_below).reshape((1,u_below.size))
         
      if(use_linear_difference):
         proposed_update = lr_input_pyr*(self.uP - vB_hat).dot(input_rate)
      else:
         proposed_update = lr_input_pyr*(soma_rate - basal_rate).dot(input_rate)
      self.input_to_pyrB_delta = low_pass_filter(self.input_to_pyrB_delta,proposed_update,low_pass_filter_timescale_weights,timestep)
      
      self.input_to_pyrB += timestep*self.input_to_pyrB_delta
      
   def update_pyr_to_int_weights(self,lr_pyr_int):
      
      soma_rate = non_linearity(self.uI)
      vI_hat     = self.gB/(self.glk + self.gD)*self.vI
      dend_rate = non_linearity(vI_hat)
      
      if(use_abs):
         pyr_rate  = np.abs(self.uP).reshape((1,self.N_pyr))
      else:
         pyr_rate  = non_linearity(self.uP).reshape((1,self.N_pyr))
         

      if(use_linear_difference):
         proposed_update = lr_pyr_int*(self.uI - vI_hat).dot(pyr_rate)
      else:
         proposed_update = lr_pyr_int*(soma_rate - dend_rate).dot(pyr_rate)
      self.pyr_to_int_delta = low_pass_filter(self.pyr_to_int_delta,proposed_update,low_pass_filter_timescale_weights,timestep)
      
      
      self.pyr_to_int += timestep*self.pyr_to_int_delta 
   
   def update_int_to_pyr_weights(self,lr_int_pyr):
      
      #
      if(use_abs):
         soma_rate = np.abs(self.uI).reshape((1,self.N_int))
      else:
         soma_rate =  non_linearity(self.uI).reshape((1,self.N_int))

      if(use_linear_difference):
         proposed_update = lr_int_pyr*(self.vA_rest-self.vA).dot(soma_rate)
      else:
         proposed_update = lr_int_pyr*(self.vA_rest-self.vA).dot(soma_rate)
      self.int_to_pyrA_delta = low_pass_filter(self.int_to_pyrA_delta,proposed_update,low_pass_filter_timescale_weights,timestep)
      
      
      
      self.int_to_pyrA +=  timestep*self.int_to_pyrA_delta 
      
   def update_above_to_pyrA_weights(self,u_above,lr_above_pyr):
      

      if(use_abs):
         above_soma_rate = np.abs(u_above)
      else:
         above_soma_rate = non_linearity(u_above)
      v_hat     = np.dot(self.above_to_pyrA,above_soma_rate)
      above_soma_rate = above_soma_rate.reshape((1,self.N_above))
      
      soma_rate = non_linearity(self.uP)
      target_rate = non_linearity(v_hat)
      
      proposed_update =  lr_above_pyr*np.dot((soma_rate - target_rate),above_soma_rate)
      self.above_to_pyrA_delta = low_pass_filter(self.above_to_pyrA_delta,proposed_update,low_pass_filter_timescale_weights,timestep)
      
      self.above_to_pyrA += timestep*self.above_to_pyrA_delta
      
      
      

class output_layer(neuron):
   
   def __init__(self,N_in,N_pyr,sigma=0.1):
      super(output_layer,self).__init__(sigma)
      self.input_to_pyrB = 0.1*np.random.uniform(-1,1,(N_pyr,N_in))
      
      self.N_pyr = N_pyr
      self.N_below = N_in

      self.input_to_pyrB_delta = 0

      self.uP  = np.random.uniform(0,1,(N_pyr,1))
      
      self.input_to_pyrB_delta = 0
      self.m = 1
      self.decay = 1
      
      
      self.vB = 1
      
      self.signal_store = []
      
      
   def update_membrane_potentials(self,u_below,target_potential=np.array([None])):
   
      u_below = u_below.reshape((self.N_below,1))
      
      noiseP = self.sigma*np.random.normal(0,1,(self.N_pyr,1))
      
      I_P = np.array([0])
      if (target_potential.any() != None):
         gE = (self.g_som*(target_potential - self.E_inh)/(self.E_ex - self.E_inh)).reshape((len(target_potential),1))
         gI = (-self.g_som*(target_potential - self.E_ex)/(self.E_ex - self.E_inh)).reshape((len(target_potential),1))
         I_P = elem_mult(gE,(self.E_ex - self.uP)) + elem_mult(gI,(self.E_inh - self.uP))
         
      self.vB = voltage_clamp(self.input_to_pyrB.dot(non_linearity(u_below)).reshape((self.N_pyr,1)))
      
      self.uP += timestep*(-self.glk*self.uP + self.gB*(self.vB - self.uP)  + noiseP + I_P)
      
   def update_input_pyr_weights(self,u_below,lr_input_pyr):
   
      soma_rate  = non_linearity(self.uP)
      vB_hat     = self.gB/(self.glk+self.gB)*self.vB
      basal_rate = non_linearity(vB_hat)
   
      if(use_abs):
         input_rate = np.abs(u_below).reshape((1,u_below.size))
      else:
         input_rate = non_linearity(u_below).reshape((1,u_below.size))
         
      if(use_linear_difference):
         proposed_update = self.input_to_pyrB_delta*(1-self.decay) + self.m*lr_input_pyr*(self.uP - vB_hat).dot(input_rate)
      else:
         proposed_update = self.input_to_pyrB_delta*(1-self.decay) + self.m*lr_input_pyr*(soma_rate - basal_rate).dot(input_rate)
      self.input_to_pyrB_delta = low_pass_filter(self.input_to_pyrB_delta,proposed_update,low_pass_filter_timescale_weights,timestep)
      
      
      self.input_to_pyrB += timestep*self.input_to_pyrB_delta