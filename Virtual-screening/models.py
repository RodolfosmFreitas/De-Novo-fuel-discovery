#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 08:18:02 2023

@author: rodolfofreitas

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch


class net_(nn.Module):
    """
    An implementation of a Fully-Connected Neural Network (Pytorch)
    """
    def __init__(self, inp_dim=5, 
                 out_dim=1,
                 n_layers=3, 
                 neurons_fc = 100,
                 hidden_activation='tanh',
                 out_layer_activation=None):
        
        super(net_, self).__init__()
        self.n_layers = n_layers
        self.input_dim = inp_dim
        self.n_layers = n_layers
        self.output_dim = out_dim
        self.neurons = neurons_fc
        
        # Define the output activation function
        if out_layer_activation is None:
            self.final_layer_activation = nn.Identity()
        elif out_layer_activation == 'sigmoid':
            self.final_layer_activation = nn.Sigmoid()
        elif out_layer_activation == 'softplus':
            self.final_layer_activation = nn.Softplus()
        elif out_layer_activation == 'tanh':
            self.final_layer_activation = nn.Tanh()
            
        # Define activation function hidden layer
        if hidden_activation is None:
            self.activation = nn.Identity()
        elif hidden_activation == 'relu':
            self.activation = nn.ReLU()
        elif hidden_activation == 'prelu':
            self.activation = nn.PReLU()
        elif hidden_activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif hidden_activation == 'elu':
            self.activation = nn.ELU()
        elif hidden_activation == 'gelu':
            self.activation = nn.GELU()
        elif hidden_activation == 'tanh':
            self.activation = nn.Tanh()
        
        # define FC layers
        self.fc_net = nn.Sequential()
        # Input layer
        self.fc_net.add_module('fc_inp', nn.Linear(self.input_dim, self.neurons))
        self.fc_net.add_module('activation', self.activation)
        #self.fc_net.add_module('dropout', nn.Dropout(0.1))
        # Hidden Layers
        for i in range(self.n_layers):
            layername = 'hidden_layer{}'.format(i+1)
            self.fc_net.add_module(layername, nn.Linear(self.neurons, self.neurons// 2))
            self.fc_net.add_module('activation{}'.format(i+1), self.activation)
            #self.fc_net.add_module('dropout'.format(i+1), nn.Dropout(0.1))
            self.neurons = self.neurons // 2
            
        # output layer
        self.fc_net.add_module('fc_out', nn.Linear(self.neurons, self.output_dim))
        self.fc_net.add_module('output_activation', self.final_layer_activation)
        
        # Initialize the Net
        self.initialize_weights()
        
    # Initialize network weights and biases using Xavier initialization
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    
    def forward(self, x):
        """
        Implement the forward pass of the NN. 
        """
        x = self.fc_net(x)
        
        return x
    
    # count the number of parameters
    def num_parameters(self):
        n_params, n_hidden_layers = 0, 0 
        for name, param in self.named_parameters():
            if 'hidden_layer' in name:
                n_hidden_layers +=1
            n_params += param.numel()
        return n_params, n_hidden_layers//2 # It counts bias as a layer, so divide by 2 
    


                    
    
    

class GPRegressionModel(gpytorch.models.ExactGP):
        """
        Gaussian process model with Spectral Mixture Kernel from Wilson et al. (2013).
        
        Used for univariate data. In our case, we (train_x, train_y) is a univariate
        time series.
        
        params:
        ------
            train_x (torch.Tensor): Tensor with the index of the time series
            train_y (torch.Tensor): Tensor with the values of the time series
            likelihood (gpytorch.likelihoods): Likelihood for the problem. We use GaussianLikelihood().
            featute_extractor(torch.nn.Module): deep learning architecture for extract a latent
            space from the high-dimensional input
        """
        
        def __init__(self, train_x, train_y, likelihood, feature_extractor, args):
            
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
            
            
            # covariance model
            if args.kernel == 'RBF':
                self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=args.features,batch_shape=torch.Size([2])),batch_shape=torch.Size([2])),
                        num_dims=args.features, grid_size=100)
                

            
            if args.kernel == 'Matern':
                self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=args.features,batch_shape=torch.Size([2])),batch_shape=torch.Size([2])),
                        num_dims=args.features, grid_size=100)
            
            elif args.kernel == 'Spectral Mixture':
                self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                    gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixtures, 
                                                               ard_num_dims=args.features,batch_shape=torch.Size([2])),batch_shape=torch.Size([2])),
                                                               num_dims=args.features, grid_size=100)
                
            self.feature_extractor = feature_extractor
            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
            
            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)) 
            
        
        
class GPRegressionModel2(gpytorch.models.ExactGP):
        """
        Gaussian process model with Spectral Mixture Kernel from Wilson et al. (2013).
        
        Used for univariate data. In our case, we (train_x, train_y) is a univariate
        time series.
        
        params:
        ------
            train_x (torch.Tensor): Tensor with the index of the time series
            train_y (torch.Tensor): Tensor with the values of the time series
            likelihood (gpytorch.likelihoods): Likelihood for the problem. We use GaussianLikelihood().
            featute_extractor(torch.nn.Module): deep learning architecture for extract a latent
            space from the high-dimensional input
        """
        
        def __init__(self, train_x, train_y, likelihood, feature_extractor, args):
            
            super(GPRegressionModel2, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            
            # covariance model
            if args.kernel == 'RBF':
                self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=args.features)),
                        num_dims=args.features, grid_size=100)
            
            if args.kernel == 'Matern':
                self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=args.features)),
                        num_dims=args.features, grid_size=100)
            
            elif args.kernel == 'Spectral Mixture':
                self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                    gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixtures, 
                                                               ard_num_dims=args.features)),
                                                               num_dims=args.features, grid_size=100)
                
            self.feature_extractor = feature_extractor
            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
            
            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




