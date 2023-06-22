# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:18:04 2022

@author: Jake
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from RNN_model_class import RNN
plt.style.use(['ggplot','paper_plots']) #add 'paper_plots.mplstyle' from my github repo to your style sheets folder if you want to use it

#%% Create random integrator
seed = None
temporally_ext_inputs = False #False is for fig 3 networks, True is for fig 4 networks
symmetric = False
rnn_int = RNN(noise_std=0,temporally_ext_inputs=temporally_ext_inputs)
params_integrator = rnn_int.create_integrator(seed=seed,symmetric=symmetric)

rnn_int.input_direction = 'amp' #'amp','pers', or 'rand'
x,r,outputs = rnn_int.run_rnn(params_integrator) #simulate integrator dynamics

#%% Plot integrator dynamics
fig = plt.figure()
two_rows = gridspec.GridSpec(2,1, figure=fig, hspace=0.5)

ax0 = fig.add_subplot(two_rows[0])
rnn_int.plot_activities(x)

ax1 = fig.add_subplot(two_rows[1])
rnn_int.plot_outputs(outputs)

#%% Delay decoding
rnn_int.input_direction = 'amp'
_,r_train,_ = rnn_int.run_rnn_batch(params_integrator)
_,r_test,_ = rnn_int.run_rnn_batch(params_integrator)
decoding = rnn_int.delay_decoding(r_train,r_test)
rnn_int.plot_delay_decoding(decoding)

#%% Cross temp decoding
rnn_int.input_direction = 'amp'
_,r_train,_ = rnn_int.run_rnn_batch(params_integrator)
_,r_test,_ = rnn_int.run_rnn_batch(params_integrator)
decoding = rnn_int.cross_temp_decoding(r_train,r_test)
rnn_int.plot_cross_temp_decoding(decoding)

#%% Plot modes overlaps
rnn_int.input_direction = 'amp'
overlaps = rnn_int.overlap_modes(params_integrator)
rnn_int.plot_overlaps(overlaps)

