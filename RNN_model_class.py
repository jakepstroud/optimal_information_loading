# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:59:21 2020

@author: Jake
"""
import numpy as np
from scipy.linalg import solve_lyapunov
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable
purples = ['#4d004b','#810f7c','#88419d','#8c6bb1','#8c96c6','#9ebcda']

#%% Create RNN class
class RNN:
    def __init__(self, N = 50,T = 2000, tau = 100, dt = 1, noise_std = 0.05, 
                 temporally_ext_inputs = False):
        
        self.N = N #number of neurons
        self.scale = 1/np.sqrt(self.N) #std of initial network weights
        self.T = T #length of simulation (ms)
        self.tau = tau #time constant (ms); 100 makes networks a little easier to train than using 50 as in the original formulation (particularly for the after_go_time cost)
        self.dt = dt #euler step size (ms)
        self.dt_tau = dt/tau #dt/tau
        self.noise_std = noise_std #std of noise in neural activities
        self.temporally_ext_inputs = temporally_ext_inputs #whether to use temporally extended inputs, if flase, cue inputs are just initial conditions
        self.batch_size_analysis = 10

        if self.temporally_ext_inputs:
            self.T = 3000 #simulate for 3000 ms if using temporally extended inputs
            self.stim_on = 500
            self.stim_off = 750
            self.go_time = self.T - 500 #2500 ms         
        else: #stimulus inputs are just initial conditions
            self.stim_on = 0
            self.stim_off = 0
            self.go_time = self.T
            
        self.cost_times = {
            'cue_delay': [self.stim_on,self.go_time],
            'just_in_time': [self.stim_off+500,self.go_time],
            'after_go_time': [self.go_time,self.T]}
            
    #Create linear integrator
    def create_integrator(self,num_inputs = 2,symmetric = False,
                          seed = None):
        np.random.seed(seed) #fix random seed if given
        
        self.num_inputs = num_inputs #number of inputs (cue conditions)
        self.nonlinearity = 'linear' #string of the neural nonlinearity to use
        self.readout_nonlinearity = 'linear' #string of the readout nonlinearity
        self.symmetric = symmetric #force weight matrix to be symmetric or not
        self.input_direction = 'amp' #use the amplifying input direction as a default
        
        imag = True
        thres = 1
        while imag or thres > 0.2: #Make sure the largest eigenvalue of W is not imaginary and, for an unconstrained network, ensure that it is relatively non-normal with a low overlap between the most amplifying and persistent modes
            W = np.random.normal(0,self.scale,(self.N,self.N))     
            if self.symmetric:
                W = 0.5*(W+W.T)
            
            w,v,_,_ = self.grab_eigs_amp(W)
            W = W + (1 - np.real(w[0]))*np.eye(self.N) #Set largest eigenvalue to be 1
            
            w,v,eigenValuesQ,eigenVectorsQ = self.grab_eigs_amp(W)
            imag = np.imag(w[0]) #Make sure it is not imaginary
            
            if self.symmetric:
                thres = 0
            else:
                thres = np.abs(np.real(eigenVectorsQ[:,0].T@v[:,0]))
        
        #Create two cue inputs
        #Persistent
        inputs_pers = np.array([v[:,0],-v[:,0]]).T #equal with opposite sign
        #amplifying
        inputs_amp = np.array([eigenVectorsQ[:,0],-eigenVectorsQ[:,0]]).T #equal with opposite sign     
        #Random
        inputs_rand = np.random.normal(0,self.scale,(self.N,1))
        inputs_rand /= np.linalg.norm(inputs_rand)
        inputs_rand = np.array([inputs_rand,-inputs_rand]).T #equal with opposite sign 
        
        inputs = {
            "pers": inputs_pers,
            "amp": inputs_amp,
            "rand": inputs_rand}
        
        b = np.zeros((self.N,self.num_inputs))
        w_out = np.array([v[:,0],v[:,0]])
        b_out = 0
        
        params_integrator = {
            "inputs": inputs,
            "W": W,
            "b": b,
            "w_out": w_out,
            "b_out": b_out
            }
        
        return params_integrator
        
    #Grab eigenvectors and amplifying modes
    def grab_eigs_amp(self,W):
        w,v = np.linalg.eig(W)
        idx = w.argsort()[::-1]
        w = w[idx];v=np.real(v[:,idx])
        
        #Shift W if largest eig is greater than 1
        if np.max(np.real(w)) > 1:
            W = W + (1 - np.max(np.real(w[0])))*np.eye(self.N)
        
        #Observability Grammian
        Q = solve_lyapunov((W-1.01*np.eye(self.N)).T,-np.eye(self.N))
        
        eigenValuesQ,eigenVectorsQ = np.linalg.eig(Q)
        idx1 = eigenValuesQ.argsort()[::-1]   
        eigenValuesQ = eigenValuesQ[idx1]
        eigenVectorsQ = eigenVectorsQ[:,idx1]
        return w,v,eigenValuesQ,eigenVectorsQ
    
    def grab_eigs_amp_lin_model(self,W):
        w,v = np.linalg.eig(W)
        idx = w.argsort()[::-1]
        w = w[idx];v=np.real(v[:,idx])
        
        #Shift W if largest eig is greater than 1
        if np.max(np.real(w)) > 1:
            W = W + (1 - np.max(np.real(w[0])))*np.eye(self.lin_N)
        
        #Observability Grammian
        Q = solve_lyapunov((W-1.01*np.eye(self.lin_N)).T,-np.eye(self.lin_N))
        
        eigenValuesQ,eigenVectorsQ = np.linalg.eig(Q)
        idx1 = eigenValuesQ.argsort()[::-1]   
        eigenValuesQ = eigenValuesQ[idx1]
        eigenVectorsQ = eigenVectorsQ[:,idx1]
        return w,v,eigenValuesQ,eigenVectorsQ   
     
    #Create numpy neural activation function
    def act_fun(self,x):
        if self.nonlinearity == 'relu':
            out = np.maximum(0,x)
        elif self.nonlinearity == 'linear':
            out = x
        return out
    
    #Create readout nonlinearity
    def readout_nonlin(self,net_out):
        if self.readout_nonlinearity == 'softmax':
            out = softmax(net_out,axis=-2)
        elif self.readout_nonlinearity == 'linear':
            out = net_out
        return out
    
    #Simulate RNN dynamics in numpy
    def run_rnn(self,params,sim_noise = None, seed = None):
        np.random.seed(seed) #fix random seed if given
        
        if sim_noise == None:
            self.sim_noise = self.noise_std
        else:
            self.sim_noise = sim_noise
        
        inputs = params['inputs']
        W = params['W']
        b = params['b']
        w_out = params['w_out']
        b_out = params['b_out']
        
        if type(inputs) is dict:
            inputs = inputs[self.input_direction]
        
        x = np.zeros((self.T,self.N,self.num_inputs))
        
        #If using temporaly extended inputs
        if self.temporally_ext_inputs:
            x[0,:] = 0
            for t in range(self.T-1):
                x[t+1,:] = x[t,:] + (self.dt_tau)*(-x[t,:] + W @ self.act_fun(x[t,:]) + b) + np.sqrt(self.dt_tau)*self.sim_noise*np.random.normal(0,1,(self.N,self.num_inputs))
                
                if t > 500 and t < 750: #stimulus period
                    x[t+1,:] += (self.dt_tau)*inputs
                elif t > 2500: #response period
                    x[t+1,:] += (self.dt_tau)*np.sum(inputs,axis=1,keepdims=True) #go cue is just sum of all inputs - as in the monkey experiment
            
        #Otherwise the inputs are just the initial conditions and there is no response period
        else:
            x[0,:] = inputs
            for t in range(self.T-1):
                x[t+1,:] = x[t,:] + (self.dt_tau)*(-x[t,:] + W @ self.act_fun(x[t,:]) + b) + np.sqrt(self.dt_tau)*self.sim_noise*np.random.normal(0,1,(self.N,self.num_inputs))
        
        r = self.act_fun(x)
        outputs = self.readout_nonlin(w_out @ self.act_fun(x) + b_out) 
        return x,r,outputs
        
    #Simulate RNN dynamics in a batch in numpy
    def run_rnn_batch(self,params,seed = None):
        np.random.seed(seed) #fix random seed if given
        
        W = params['W']
        b = np.expand_dims(params['b'],axis=0)
        w_out = params['w_out']
        b_out = params['b_out']
        if np.ndim(b_out) == 0:
            b_out = b_out
        else:
            b_out = np.expand_dims(b_out,axis=1)
        
        inputs = params['inputs']
        if type(inputs) is dict:
            inputs = inputs[self.input_direction]
            
        inputs = np.expand_dims(inputs,axis=0)
        x = np.zeros((self.T,self.batch_size_analysis,self.N,self.num_inputs))
        
        #If using temporaly extended inputs
        if self.temporally_ext_inputs:
            x[0,:] = 0
            for t in range(self.T-1):
                x[t+1,:] = x[t,:] + (self.dt_tau)*(-x[t,:] + W @ self.act_fun(x[t,:]) + b) + np.sqrt(self.dt_tau)*self.noise_std*np.random.normal(0,1,(self.batch_size_analysis,self.N,self.num_inputs))
                
                if t > 500 and t < 750: #stimulus period
                    x[t+1,:] += (self.dt_tau)*inputs
                elif t > 2500: #response period
                    x[t+1,:] += (self.dt_tau)*np.sum(inputs,axis=2,keepdims=True) #go cue is just sum of all inputs - as in the monkey experiment
            
        #Otherwise the inputs are just the initial conditions and there is no response period
        else:
            x[0,:] = inputs
            for t in range(self.T-1):
                x[t+1,:] = x[t,:] + (self.dt_tau)*(-x[t,:] + W @ self.act_fun(x[t,:]) + b) + np.sqrt(self.dt_tau)*self.noise_std*np.random.normal(0,1,(self.batch_size_analysis,self.N,self.num_inputs))
        
        r = self.act_fun(x)
        outputs = self.readout_nonlin(w_out @ self.act_fun(x) + b_out) 
        return x,r,outputs
    
    def plot_activities(self,x,cue_cond=0,n_neurons_plot=10):
        plt.plot(x[:,:n_neurons_plot,cue_cond])
        if self.temporally_ext_inputs:
            plt.xticks([0,500,1000,1500,2000,2500,3000],['-500','0','500','1000','1500','2000','2500'])
        plt.ylabel('neural acitivity')
        plt.xlabel('time (ms)')
        
    def plot_outputs(self,outputs):
        # if self.readout_nonlinearity == 'linear':
        #     outputs*=np.sign(outputs[-1,0,0]) #sign flip so both readouts are positive
        [plt.plot(outputs[:,i,i],clip_on=False,label='cue '+str(i+1)) for i in range(self.num_inputs)]
        # plt.text(self.T/3,0.35,'chance decoding',color=[0.5,0.5,0.5])
        if self.T == 3000:
            plt.xticks([0,500,1000,1500,2000,2500,3000],['-500','0','500','1000','1500','2000','2500'])
        plt.xlabel('time (ms)');plt.ylabel('readouts')
        plt.legend(ncol=3)
        if self.readout_nonlinearity == 'softmax':
            plt.plot([0,self.T],[1/self.num_inputs,1/self.num_inputs],'--',color=[0.5,0.5,0.5],label='chance')
            plt.ylim([0,1])
        
    #Initialise network parameters before training
    def initialise_params(self,num_inputs = 6,nonlinearity = 'relu',symmetric = False,
                          readout_nonlinearity = 'softmax',seed = None):
        np.random.seed(seed) #fix random seed if given        
        
        self.num_inputs = num_inputs #number of inputs (cue conditions)
        self.nonlinearity = nonlinearity #'relu' or 'linear'; string of the neural nonlinearity to use
        self.symmetric = symmetric #force weight matrix to be symmetric or not
        self.readout_nonlinearity = readout_nonlinearity #'softmax' or 'linear'; string of the readout nonlinearity
        
        inputs = np.random.normal(0,self.scale,(self.N,self.num_inputs))
        W = np.random.normal(0,self.scale,(self.N,self.N))
        if self.symmetric:
            W = 0.5*(W+W.T)
        b = np.random.normal(0,self.scale,(self.N,self.num_inputs))
        w_out = np.random.normal(0,self.scale,(self.num_inputs,self.N))
        b_out = np.random.normal(0,self.scale,(1,self.num_inputs,1))
        
        params_init = {
            "inputs": inputs,
            "W": W,
            "b": b,
            "w_out": w_out,
            "b_out": b_out}
        
        return params_init
    
    #Simulate fitted linear models dynamics
    def run_fitted_lin_model(self,params):        
        
        inputs = params['inputs']
        W = params['W']
        b = params['b']
        w_out = params['w_out']
        
        if type(inputs) is dict:
            inputs = inputs[self.input_direction]
        
        x = np.zeros((self.T_lin_model,self.lin_N,self.num_inputs))
        
        #If using temporaly extended inputs
        if self.temporally_ext_inputs:
            x[0,:] = 0
            for t in range(self.T_lin_model-1):
                x[t+1,:] = x[t,:] + (self.dt_tau)*(-x[t,:] + W @ x[t,:] + b)
                
                if t < 250: #stimulus period
                    x[t+1,:] += (self.dt_tau)*inputs

        #Otherwise the inputs are just the initial conditions and there is no response period
        else:
            x[0,:] = inputs
            for t in range(self.T_lin_model-1):
                x[t+1,:] = x[t,:] + (self.dt_tau)*(-x[t,:] + W @ x[t,:] + b)
        
        outputs = w_out @ x
        return x,outputs
    
    #%% Tensorflow stuff
    @tf.function
    def act_fun_tf(self,x_tf):
        if self.nonlinearity == 'relu':
            out = tf.nn.relu(x_tf)
        elif self.nonlinearity == 'linear':
            out = x_tf
        return out
    
    @tf.function
    def cost_fun(self,outputs_tf): #Cross-entropy loss
        return -tf.reduce_sum(tf.linalg.trace(tf.math.log(tf.nn.softmax(outputs_tf,axis=1))))

    @tf.function
    def cond(self,x_tf,t,inputs_tf,W_tf,b_tf,w_out_tf,b_out_tf,cost):
        return tf.less(t,self.T)
    
    @tf.function
    def body(self,x_tf,t,inputs_tf,W_tf,b_tf,w_out_tf,b_out_tf,cost):
        
        #Run rnn dynamics
        if self.temporally_ext_inputs:            
            x_tf = x_tf + self.dt_tau * (-x_tf + W_tf @ self.act_fun_tf(x_tf) + b_tf) + np.sqrt(self.dt_tau)*self.noise_std*tf.random.normal((self.batch_size,self.N,self.num_inputs),dtype=tf.float64)
            
            #Stimulus period
            x_tf = tf.cond(tf.logical_and(t >= self.stim_on,t < self.stim_off),
                           lambda: x_tf + self.dt_tau*inputs_tf, lambda: x_tf) #apply the inputs during the stimulus period
            
            #Response time
            x_tf = tf.cond(t > self.go_time, lambda: x_tf + self.dt_tau*(
                tf.expand_dims(tf.reduce_sum(inputs_tf,axis=-1),axis=-1)), lambda: x_tf)
            
        else:
            x_tf = tf.cond(t==0,lambda: inputs_tf, lambda: x_tf) #set initial condition to be the inputs
            x_tf = x_tf + self.dt_tau * (-x_tf + W_tf @ self.act_fun_tf(x_tf) + b_tf) + np.sqrt(self.dt_tau)*self.noise_std*tf.random.normal((self.batch_size,self.N,self.num_inputs),dtype=tf.float64)
        r_tf = self.act_fun_tf(x_tf)
        
        #Apply rate regularisation at all times
        cost += self.l2_reg*tf.nn.l2_loss(r_tf)
                
        #Apply cross entropy loss using the provided cost type (either just_in_time, cue_delay, or after_go_time)
        cost = tf.cond(tf.logical_and(t >= self.cost_start,t < self.cost_stop), 
        lambda:cost+(self.cost_scale/self.cost_time_length)*self.cost_fun(w_out_tf@r_tf + b_out_tf),
        lambda:cost)
        
        return x_tf,t+1,inputs_tf,W_tf,b_tf,w_out_tf,b_out_tf,cost
    
    @tf.function
    def run_rnn_while(self,inputs_tf,W_tf,b_tf,w_out_tf,b_out_tf):
        cost = tf.constant(0.0,dtype=tf.float64)
        t = tf.constant(0,dtype=tf.int64)
        x_tf = tf.zeros(inputs_tf.shape,dtype=tf.float64)
        
        x_tf = tf.stack([x_tf]*self.batch_size)                
        inputs_tf = tf.stack([inputs_tf]*self.batch_size)
        b_tf = tf.stack([b_tf]*self.batch_size)
                
        if self.symmetric:
            W_tf = 0.5*(W_tf+tf.transpose(W_tf))
            
        x_tf,t,inputs_tf,W_tf,b_tf,w_out_tf,b_out_tf,cost = tf.while_loop(
            self.cond, self.body, (x_tf,t,inputs_tf,W_tf,b_tf,w_out_tf,b_out_tf,cost), parallel_iterations=10)
        return cost
    
    def initialise_tf_params(self,params_init):       
        inputs_tf = tf.Variable(params_init['inputs'])
        W_tf = tf.Variable(params_init['W'])
        b_tf = tf.Variable(params_init['b'])
        w_out_tf = tf.Variable(params_init['w_out'])
        b_out_tf = tf.Variable(params_init['b_out'])
        
        return inputs_tf,W_tf,b_tf,w_out_tf,b_out_tf
        
    def train_rnn(self,params_init,cost_type = 'just_in_time',batch_size = 10,
                  learning_rate = 0.005,l2_reg = 0.0005,cost_scale=1000,
                  n_train_steps = 2000,seed = None):
        
        tf.random.set_seed(0)
        self.cost_type = cost_type #'cue_delay','just_in_time','after_go_time'; cost function used to train the network
        self.batch_size = batch_size #batch size
        self.learning_rate = learning_rate #learning rate
        self.l2_reg = l2_reg #stength of L2 regularisation on firing rates during training
        self.cost_scale = cost_scale #scaling the weight of the cross-entropy loss function
        self.n_train_steps = n_train_steps #number of training steps (200 may be enough, but 1000 may be needed for complete convergence)
        cost_over_training = tf.TensorArray(size = self.n_train_steps,dtype=tf.float64)        
        
        if self.cost_type == 'after_go_time' and self.temporally_ext_inputs == False:
            raise Exception('Cannot use this cost function without a response window, please use temporally extended inputs or one of the other cost functions.')
        
        [self.cost_start,self.cost_stop] = self.cost_times[cost_type]
        self.cost_time_length = self.cost_stop - self.cost_start
        
        inputs_tf,W_tf,b_tf,w_out_tf,b_out_tf = self.initialise_tf_params(params_init)
        
        #Force to CPU
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')        
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)      
        
        #Train
        for epoch in range(self.n_train_steps):
            with tf.GradientTape() as tape:
                cost = self.run_rnn_while(inputs_tf,W_tf,b_tf,w_out_tf,b_out_tf)
                
            gradients = tape.gradient(cost,[inputs_tf,W_tf,b_tf,w_out_tf,b_out_tf])    
            optimizer.apply_gradients(zip(gradients,[inputs_tf,W_tf,b_tf,w_out_tf,b_out_tf]))        
            cost_over_training.write(epoch,cost) #Grab cost over time
    
            if epoch%10==0:
                print('Num. steps:',epoch,', cost:',int(cost.numpy()))
        
        cost_over_training = cost_over_training.stack()
        self.cost_over_training = cost_over_training.numpy()
        
        #Explicity convert to numpy arrays
        params_trained = {
            "inputs": np.squeeze(inputs_tf.numpy()),
            "W": W_tf.numpy(),
            "b": b_tf.numpy(),
            "w_out": w_out_tf.numpy(),
            "b_out": b_out_tf.numpy()}
        
        return params_trained
    
    def fit_lin_model(self,rates,t_start=500,t_end=1500,
                  learning_rate = 0.0001,l2_reg = 1/12,
                  n_train_steps = 10000,seed = None):        
        
        self.pca_reduction(rates,t_start,t_end)
        params_trained = self.train_rnn_lin_model(learning_rate,l2_reg,n_train_steps,seed)
        
        return params_trained
    
    def pca_reduction(self,rates,t_start,t_end):
        
        self.lin_N = 20
        
        #PCA reduction
        rates = np.mean(rates,axis=1)
        rates -= np.mean(rates,axis=-1,keepdims=True)
        
        rates = np.transpose(rates,[0,2,1])
        pca = PCA(n_components=self.lin_N)
        self.T_lin_model = t_end-t_start
        
        pca.fit(np.reshape(rates[t_start:t_end,:],(self.T_lin_model*self.num_inputs,self.N)))
        r_pc = rates[t_start:t_end,:]@pca.components_.T
        self.r_pc = np.transpose(r_pc,[0,2,1])
        
        pc_weights = np.expand_dims(pca.explained_variance_ratio_,axis=1).T
        pc_weights /= np.sum(pc_weights)
        self.pc_weights = pc_weights
    
    def initialise_params_lin_model(self):
        W = np.random.normal(0, 1/np.sqrt(self.lin_N), (self.lin_N,self.lin_N))
        b = np.random.normal(0, 1/np.sqrt(self.lin_N), (self.lin_N,self.num_inputs))
        inputs = np.random.normal(0, 1/np.sqrt(self.lin_N), (self.lin_N,self.num_inputs))
        w_out = np.random.normal(0, 1/np.sqrt(self.lin_N), (self.lin_N,self.lin_N))
        
        params_init = {
            "inputs": inputs,
            "W": W,
            "b": b,
            "w_out": w_out}
        
        return params_init
    
    @tf.function
    def get_cost_lin_model(self,cost_in):
        return tf.reduce_mean(self.pc_weights_tf @ tf.pow(cost_in,2)) #tf.reduce_mean(self.pc_weights_tf @ tf.pow(cost_in,2))

    @tf.function
    def cond_lin_model(self,x_tf,t,inputs_tf,W_tf,b_tf,w_out_tf,cost):
        return tf.less(t,self.T_lin_model)
    
    @tf.function
    def body_lin_model(self,x_tf,t,inputs_tf,W_tf,b_tf,w_out_tf,cost):
        
        #Run rnn dynamics
        if self.temporally_ext_inputs:            
            x_tf = x_tf + self.dt_tau * (-x_tf + W_tf @ x_tf + b_tf)
            
            #Stimulus period
            x_tf = tf.cond(t < 250,
                           lambda: x_tf + self.dt_tau*inputs_tf, lambda: x_tf) #apply the inputs during the stimulus period
            
        else:
            # x_tf = tf.cond(t==0,lambda: inputs_tf, lambda: x_tf) #set initial condition to be the inputs
            x_tf = x_tf + self.dt_tau * (-x_tf + W_tf @ x_tf + b_tf)
        
        cost += self.get_cost_lin_model(w_out_tf@x_tf - self.r_pc_tf[t,:,:])/self.T_lin_model_tf
        
        return x_tf,t+1,inputs_tf,W_tf,b_tf,w_out_tf,cost
    
    @tf.function
    def run_rnn_while_lin_model(self,inputs_tf,W_tf,b_tf,w_out_tf):
        cost = tf.constant(0.0,dtype=tf.float64)
        t = tf.constant(0,dtype=tf.int64)
        
        if self.temporally_ext_inputs:
            x_tf = tf.zeros(inputs_tf.shape,dtype=tf.float64)
        else:
            x_tf = inputs_tf
        
        self.r_pc_tf = tf.constant(self.r_pc,dtype=tf.float64)   
        self.pc_weights_tf = tf.constant(self.pc_weights,dtype=tf.float64)
        self.T_lin_model_tf = tf.constant(self.T_lin_model,dtype=tf.float64)
   
        x_tf,t,inputs_tf,W_tf,b_tf,w_out_tf,cost = tf.while_loop(
            self.cond_lin_model, self.body_lin_model, (x_tf,t,inputs_tf,W_tf,b_tf,w_out_tf,cost), parallel_iterations=10)
        
        #Add reg:
        cost += self.l2_reg_lin_model*(tf.nn.l2_loss(b_tf)+(1/self.num_inputs)*tf.nn.l2_loss(inputs_tf)+(1/self.lin_N)*tf.nn.l2_loss(w_out_tf))
        
        return cost
    
    def initialise_tf_params_lin_model(self,params_init):       
        inputs_tf = tf.Variable(params_init['inputs'])
        W_tf = tf.Variable(params_init['W'])
        b_tf = tf.Variable(params_init['b'])
        w_out_tf = tf.Variable(params_init['w_out'])
        
        return inputs_tf,W_tf,b_tf,w_out_tf
     
    def train_rnn_lin_model(self,
              learning_rate = 0.0001,l2_reg = 1/12,
              n_train_steps = 10000,seed = None):
        
        tf.random.set_seed(0)
        self.learning_rate_lin_model = learning_rate #learning rate
        self.l2_reg_lin_model = l2_reg #stength of L2 regularisation on firing rates during training
        self.n_train_steps_lin_model = n_train_steps #number of training steps (200 may be enough, but 1000 may be needed for complete convergence)
        cost_over_training = tf.TensorArray(size = self.n_train_steps_lin_model,dtype=tf.float64)        
        
        params_init = self.initialise_params_lin_model()
        inputs_tf,W_tf,b_tf,w_out_tf = self.initialise_tf_params_lin_model(params_init)
        
        #Force to CPU
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')        
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate_lin_model)      
        
        #Train
        for epoch in range(self.n_train_steps_lin_model):
            with tf.GradientTape() as tape:
                cost = self.run_rnn_while_lin_model(inputs_tf,W_tf,b_tf,w_out_tf)
                
            gradients = tape.gradient(cost,[inputs_tf,W_tf,b_tf,w_out_tf])    
            optimizer.apply_gradients(zip(gradients,[inputs_tf,W_tf,b_tf,w_out_tf]))        
            cost_over_training.write(epoch,cost) #Grab cost over time
    
            if epoch%100==0:
                print('Num. steps:',epoch,', cost:',np.round(cost.numpy(),4))
        
        cost_over_training = cost_over_training.stack()
        self.cost_over_training_lin_model = cost_over_training.numpy()
        
        #Explicity convert to numpy arrays
        params_trained = {
            "inputs": np.squeeze(inputs_tf.numpy()),
            "W": W_tf.numpy(),
            "b": b_tf.numpy(),
            "w_out": w_out_tf.numpy()}
        
        return params_trained       
    
    #%% Analysis methods
    def cross_temp_decoding(self,r_train,r_test):
        
        r_train = np.transpose(r_train,[0,1,3,2])
        r_test = np.transpose(r_test,[0,1,3,2])
        
        lin_model = LogisticRegression(C=0.5)
        targets = (np.stack([np.arange(self.num_inputs)]*self.batch_size_analysis)).flatten()
        num_times = int(self.T/10)
        decoding = np.zeros((num_times,num_times))        
        
        for t1 in range(num_times):
            
            r_train_shaped = np.reshape(r_train[10*t1,:],(self.batch_size_analysis*self.num_inputs,self.N))
            lin_model.fit(r_train_shaped,targets)
            for t2 in range(num_times):
                r_test_shaped = np.reshape(r_test[10*t2,:],(self.batch_size_analysis*self.num_inputs,self.N))
                decoding[t1,t2] = lin_model.score(r_test_shaped,targets)
                
        return decoding
    
    def plot_cross_temp_decoding(self,decoding):
        
        im = plt.imshow(decoding,origin='lower',vmin=0,vmax=1)
        if self.temporally_ext_inputs:
            plt.xticks([0,50,100,150,200,250,300],['-500','0','500','1000','1500','2000','2500'])
            plt.yticks([0,50,100,150,200,250,300],['-500','0','500','1000','1500','2000','2500'])
            plt.plot([self.stim_on/10,self.stim_on/10],[0,300],'-w')
            plt.plot([0,300],[self.stim_on/10,self.stim_on/10],'-w')
            plt.plot([self.stim_off/10,self.stim_off/10],[0,300],'-w')
            plt.plot([0,300],[self.stim_off/10,self.stim_off/10],'-w')
            plt.plot([self.go_time/10,self.go_time/10],[0,300],'-w')
            plt.plot([0,300],[self.go_time/10,self.go_time/10],'-w')
        else:
            plt.xticks([0,50,100,150,200],['0','500','1000','1500','2000'])
            plt.yticks([0,50,100,150,200],['0','500','1000','1500','2000'])
        
        plt.ylim([0,self.T/10])
        plt.xlim([0,self.T/10])
        plt.ylabel('train time (ms)')
        plt.xlabel('test time (ms)')
        self.colorbar(im,'decoding accuracy')
       
    def colorbar(self,mappable,label):
        fig = plt.gcf()
        last_axes = plt.gca()
        ax = mappable.axes
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.07, pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(label, rotation=270, labelpad=10)
        plt.sca(last_axes)
        return cbar

    def delay_decoding(self,r_train,r_test):
        
        r_train = np.transpose(r_train,[0,1,3,2])
        r_test = np.transpose(r_test,[0,1,3,2])
        
        lin_model = LogisticRegression(C=0.5)
        targets = (np.stack([np.arange(self.num_inputs)]*self.batch_size_analysis*500)).flatten()
        targets_test = (np.stack([np.arange(self.num_inputs)]*self.batch_size_analysis)).flatten()

        num_times = int(self.T/10)
        decoding = np.zeros(num_times)        
        
        r_train_shaped = np.reshape(r_train[self.go_time-500:self.go_time,:],(self.batch_size_analysis*self.num_inputs*500,self.N))
        lin_model.fit(r_train_shaped,targets)
        
        for t1 in range(num_times):            
            
            r_test_shaped = np.reshape(r_test[10*t1,:],(self.batch_size_analysis*self.num_inputs,self.N))
            decoding[t1] = lin_model.score(r_test_shaped,targets_test)
                
        return decoding
    
    def plot_delay_decoding(self,decoding):
        
        plt.plot(decoding,clip_on=False)
        if self.temporally_ext_inputs:
            plt.xticks([0,50,100,150,200,250,300],['-500','0','500','1000','1500','2000','2500'])
            plt.plot([self.stim_on/10,self.stim_on/10],[0,1],'-y')
            plt.plot([self.stim_off/10,self.stim_off/10],[0,1],'-y')
            plt.plot([self.go_time/10,self.go_time/10],[0,1],'-y')
        else:
            plt.xticks([0,50,100,150,200],['0','500','1000','1500','2000'])
        
        plt.ylim([0,1])
        plt.xlim([0,self.T/10])
        plt.xlabel('time (ms)')   
        plt.ylabel('decoding accuracy')
    
    def overlap_modes(self,params):
        
        n_pcs = int(self.N/4)
        step = 20
        pca = PCA(n_components=n_pcs)
        x,r,outputs = self.run_rnn(params,sim_noise=0)
        r = r - np.mean(r,axis=-1,keepdims=True)
        
        if self.nonlinearity == 'relu':
            w,v,eigenValuesQ,eigenVectorsQ = self.grab_eigs_amp(0.5*params['W'])
        else:
            w,v,eigenValuesQ,eigenVectorsQ = self.grab_eigs_amp(params['W'])        
        v,_ = np.linalg.qr(v) #orthogonalize the eigenvectors
        
        v_rand = np.random.normal(0,1,(200,self.N,self.N))
        for i in range(200):
            v_rand[i,:],_ = np.linalg.qr(v_rand[i,:])
        
        if self.T == 3000:
            self.overlap_times = np.linspace(500,self.T-500-step,5).astype(int)
        else:
            self.overlap_times = np.linspace(0,self.T-step-1,5).astype(int)
        self.overlap_times_plot = np.linspace(0,self.T-step-1,5).astype(int)
            
        overlaps = np.zeros((5,2)) #time * persistent and amplifying
        overlaps_random = np.zeros((5,200))
        
        for t_c,t in enumerate(self.overlap_times):
            
            r_ = np.transpose(r[t:t+step,:],[0,2,1]).reshape((step*self.num_inputs,self.N))
            pca.fit(r_)
            C = r_.T@r_
            tot_var = np.trace(pca.components_@C@pca.components_.T)
            
            overlaps[t_c,0] = np.trace((v.T@C@v)[:n_pcs,:n_pcs])/tot_var
            overlaps[t_c,1] = np.trace((eigenVectorsQ.T@C@eigenVectorsQ)[:n_pcs,:n_pcs])/tot_var
            overlaps_random[t_c,:] = np.trace((np.transpose(v_rand,[0,2,1])@C@v_rand)[:,:n_pcs,:n_pcs],axis1=-2,axis2=-1)/tot_var
            
        return [100*overlaps,100*overlaps_random]
        
    def plot_overlaps(self,overlaps):
        
        plt.plot(self.overlap_times_plot,overlaps[0][:,0],'.-g',label='persistent',clip_on=False)
        plt.plot(self.overlap_times_plot,overlaps[0][:,1],'.-r',label='amplifying',clip_on=False)
        
        plt.plot(self.overlap_times_plot,np.mean(overlaps[1],axis=1),'.-k',label='random',clip_on=False)
        plt.fill_between(self.overlap_times_plot, y1 = np.percentile(overlaps[1],2.5,axis=1), 
                         y2 = np.percentile(overlaps[1],97.5,axis=1),color = 'k',alpha=0.5,linewidth=0)
        
        plt.ylim([0,100])
        plt.xlabel('time from cue onset (ms)')
        plt.ylabel('variance explained (%)')
        plt.legend()
        
    def local_lin_input_modes(self,params,n_pcs = None):
        
        inputs = params['inputs']
        if n_pcs == None:
            n_pcs = self.num_inputs
        pca = PCA(n_components=n_pcs)
        
        if self.nonlinearity == 'relu':
            w,v,eigenValuesQ,eigenVectorsQ = self.grab_eigs_amp(0.5*params['W'])
        else:
            w,v,eigenValuesQ,eigenVectorsQ = self.grab_eigs_amp(params['W'])        
        v,_ = np.linalg.qr(v) #orthogonalize the eigenvectors
        
        v_rand = np.random.normal(0,1,(200,self.N,self.N))
        for i in range(200):
            v_rand[i,:],_ = np.linalg.qr(v_rand[i,:])
        
        C = inputs@inputs.T
        pca.fit(inputs.T)
        tot_var = np.trace(pca.components_@C@pca.components_.T)
        
        pers_overlap = np.trace((v.T@C@v)[:n_pcs,:n_pcs])/tot_var
        amp_overlap = np.trace((eigenVectorsQ.T@C@eigenVectorsQ)[:n_pcs,:n_pcs])/tot_var
        rand_overlap = np.trace((np.transpose(v_rand,[0,2,1])@C@v_rand)[:,:n_pcs,:n_pcs],axis1=-2,axis2=-1)/tot_var
        
        return [pers_overlap,amp_overlap,rand_overlap]
    
    def plot_input_mode_overlaps(self,pers_overlap,amp_overlap,rand_overlap):
        
        plt.plot(0,pers_overlap,'.g')
        plt.plot(1,amp_overlap,'.r')
        m = np.mean(rand_overlap);s = np.std(rand_overlap)
        plt.errorbar(2, y=m, yerr=s,color='k',fmt='.-')
        
        plt.xlim([-0.5,2.5])
        plt.ylim([0,1])
        plt.ylabel('overlap of inputs with modes')
        plt.xticks([0,1,2],['persistent\nmodes','amplifying\nmodes','random\nmodes'])        
        
    def plot_top_pc_fitted_lin_model(self,params_trained_lin_model,pc=0):
        
        x,outputs = self.run_fitted_lin_model(params_trained_lin_model)
        [plt.plot(self.r_pc[:,pc,i],'-',color=purples[i]) for i in range(self.num_inputs)]
        [plt.plot(outputs[:,pc,i],'--',color=purples[i]) for i in range(self.num_inputs)]

    def overlap_modes_lin_model(self,params):
        
        n_pcs = int(self.lin_N/4)
        step = 20
        pca = PCA(n_components=n_pcs)
        r,outputs = self.run_fitted_lin_model(params)
        
        w,v,eigenValuesQ,eigenVectorsQ = self.grab_eigs_amp_lin_model(params['W'])        
        v,_ = np.linalg.qr(v) #orthogonalize the eigenvectors
        
        v_rand = np.random.normal(0,1,(200,self.lin_N,self.lin_N))
        for i in range(200):
            v_rand[i,:],_ = np.linalg.qr(v_rand[i,:])
        
        self.overlap_times = np.linspace(0,self.T_lin_model-step,5).astype(int)        
        self.overlap_times_plot = self.overlap_times
        
        overlaps = np.zeros((5,2)) #time * persistent and amplifying
        overlaps_random = np.zeros((5,200))
        
        for t_c,t in enumerate(self.overlap_times):
            
            r_ = np.transpose(r[t:t+step,:],[0,2,1]).reshape((step*self.num_inputs,self.lin_N))
            pca.fit(r_)
            C = r_.T@r_
            tot_var = np.trace(pca.components_@C@pca.components_.T)
            
            overlaps[t_c,0] = np.trace((v.T@C@v)[:n_pcs,:n_pcs])/tot_var
            overlaps[t_c,1] = np.trace((eigenVectorsQ.T@C@eigenVectorsQ)[:n_pcs,:n_pcs])/tot_var
            overlaps_random[t_c,:] = np.trace((np.transpose(v_rand,[0,2,1])@C@v_rand)[:,:n_pcs,:n_pcs],axis1=-2,axis2=-1)/tot_var
            
        return [100*overlaps,100*overlaps_random]


