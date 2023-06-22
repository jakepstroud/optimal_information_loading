# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:59:21 2020

@author: Jake
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from RNN_model_class import RNN
plt.style.use(['ggplot','paper_plots']) #add 'paper_plots.mplstyle' from my github repo to your style sheets folder if you want to use it

#%% Initialise the params of a nonlinear RNN
seed = None
temporally_ext_inputs = False #False is for fig 2 networks, True is for fig 6 networks
symmetric = False
rnn = RNN(temporally_ext_inputs = temporally_ext_inputs)
params_init = rnn.initialise_params(seed=seed,symmetric=symmetric)
x,r,outputs = rnn.run_rnn(params_init,seed=seed)

#%% Plot initial dynamics prior to training
fig = plt.figure()
two_rows = gridspec.GridSpec(2,1, figure=fig, hspace=0.5)

ax0 = fig.add_subplot(two_rows[0])
rnn.plot_activities(x)

ax1 = fig.add_subplot(two_rows[1])
rnn.plot_outputs(outputs)

#%% Train the network
tic = time.time()

cost_type = 'just_in_time' #'cue_delay', 'just_in_time', or 'after_go_time'
params_trained = rnn.train_rnn(params_init,seed=seed,cost_type = cost_type,l2_reg = 0.0005)
#l2_reg may need a little tuning sometimes depending on the set up you wish to use.  

toc = time.time()
print('time taken:', np.round((toc-tic)/60,2),'mins')

plt.plot(rnn.cost_over_training) #plot training loss

#%% Plot dynamics after training
x_trained,r_trained,outputs_trained = rnn.run_rnn(params_trained,seed=seed) #Grab dynamics after training

fig = plt.figure()
two_rows = gridspec.GridSpec(2,1, figure=fig, hspace=0.5)

ax0 = fig.add_subplot(two_rows[0])
rnn.plot_activities(x_trained)

ax1 = fig.add_subplot(two_rows[1])
rnn.plot_outputs(outputs_trained)

#%% Plot overlap of optimized inputs with modes based on local linearization
pers_overlap,amp_overlap,rand_overlap = rnn.local_lin_input_modes(params_trained)
rnn.plot_input_mode_overlaps(pers_overlap,amp_overlap,rand_overlap)

#%% Cross temp decoding
_,r_train,_ = rnn.run_rnn_batch(params_trained)
_,r_test,_ = rnn.run_rnn_batch(params_trained)
decoding = rnn.cross_temp_decoding(r_train,r_test)
rnn.plot_cross_temp_decoding(decoding)

#%% Fit linear model to nonlinear dynamics
x,r,outputs = rnn.run_rnn_batch(params_trained,seed=seed)
params_trained_lin_model = rnn.fit_lin_model(r)

#%% Simulate fitted linear model
rnn.plot_top_pc_fitted_lin_model(params_trained_lin_model,pc=0)

#%% Plot overlaps of fitted linear model
overlaps = rnn.overlap_modes_lin_model(params_trained_lin_model)
rnn.plot_overlaps(overlaps)

