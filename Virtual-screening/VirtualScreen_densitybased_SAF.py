# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:55:36 2024

@author: Rodolfo Freitas
"""

import os
import pandas as pd
import numpy as np
from models import net_, GPRegressionModel
import deepchem as dc
import torch 
import gpytorch
from datetime import datetime
from timeit import default_timer
from math import ceil, floor

from scipy.optimize import minimize
from scipy.optimize import Bounds
from joblib import Parallel, delayed
import multiprocessing
import argparse


# Train
parser = argparse.ArgumentParser(description='Deep Kernel learning + Molecular representation')

parser.add_argument('--data-dir', type=str, default="../data", help='data directory')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learnign rate')

# Feature extractor (NN)
parser.add_argument('--num-layers', type=int, default=3, help='number of FC layers')
parser.add_argument('--neurons-fc', type=int, default=256, help='number of neurons in the first layer of nn')
parser.add_argument('--features', type=int, default=2, help='number of features')
parser.add_argument('--activation', type=str, default='elu', help='Hidden layer activation, [relu, elu, gelu, tanh, None=linear]')
parser.add_argument('--output-activation', type=str, default=None, help='Output layer activation, sigmoid ~ [0,1], None=linear, softplus ~ [0, inf[')

# Kernel
parser.add_argument('--kernel', type=str, default='RBF', help='Covariance model = [RBF, Matern, Spectral Mixture]')
parser.add_argument('--num_mixtures', type=int, default=2, help='number of mixtures in Spectral Mixture model')
args = parser.parse_args()


#

def predict(test_x):
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Test points are regularly spaced along [0,1]
        Y_pred = likelihood(model(test_x))

    # Sampling from the posterior
    f_samples = Y_pred.sample(sample_shape=torch.Size([1000])).detach().cpu().numpy() * std + mu
    
    mu_pred = f_samples.mean(0) 
    var_pred = Y_pred.var(0)
    sigma_pred = torch.sqrt(var_pred)
    
    return mu_pred, sigma_pred


def jac_fun(x, phi, y_true, alpha):
    
    phi = torch.tensor(phi, dtype=torch.float, requires_grad=True).to(device)
    x = torch.tensor(x, dtype=torch.float, requires_grad=True).to(device)
    
    model.eval()
    likelihood.eval()
    Y_pred = likelihood(model(torch.mm(x.unsqueeze(0),phi))).mean * std + mu 
    jac = torch.autograd.grad(Y_pred.sum(), x)   
    return jac[0].detach().cpu().numpy()
    
def obj_fun(x, phi, y_true, alpha):
    # Mixing Operator
    
    Phi = torch.tensor(np.matmul(x[None,:], phi), dtype=torch.float).to(device)
    
    # call the surrogate model
    mu, sigma = predict(Phi)
    
    '''
    loss = (y_true - mu)^2 + beta * (sigma - |y_true - mu|)^2 + alpha * |omega|
    '''
    beta = 1e-2
    var_reg = beta*np.mean(np.square(sigma.detach().cpu().numpy()[0][0] - np.absolute(mu.detach().cpu().numpy()[0][0] - y_true)))
    # Lasso
    l1_reg = np.linalg.norm(x, 1)
    
    
    loss = np.mean(np.square(y_true - mu.detach().cpu().numpy()[0][0])) + var_reg +  alpha * l1_reg 
    
    return loss

def opt(k, alpha, target):
    # Constrains x \in [0, 1] & \sum(x) = 1.0
    bounds = Bounds([1]*np.zeros(phi.shape[0]), [1]*np.ones(phi.shape[0]))
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.}
    
    # Initial condition (sampling from a sparse dirichlet distribution) 
    x0 = np.random.dirichlet(np.ones(phi.shape[0]))

    options = {'maxiter': 1000, 'ftol':1e-9}
    sol = minimize(obj_fun, 
                   x0, 
                   method='SLSQP',
                   jac=jac_fun,
                   bounds=bounds,
                   constraints=[cons],
                   options=options,
                   args=(phi, target, alpha))
    
    x = sol.x 
    objective_funtion = sol.fun
    return x, objective_funtion
    

if __name__ == "__main__":
    date_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    #%% READ MOLECULES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read fuel data [Name, SMILES, CN, METHOD]
    data = pd.read_excel('{}/fuel_palette_SAF.xlsx'.format(args.data_dir)).to_numpy() 
    fuel_list = data[:,0]
    SMILES = data[:,1].tolist() 
    classes = data[:,-1].tolist() 

    #%% Mol2Vec Fingerprinters
    '''
    Jaeger, Sabrina, Simone Fulle, and Samo Turk. 
    “Mol2vec: unsupervised machine learning approach with chemical intuition.” 
    Journal of chemical information and modeling 58.1 (2018): 27-35.
    Pre-trained model from https://github.com/samoturk/mol2vec/
    The default model was trained on 20 million compounds downloaded from ZINC 
    using the following paramters.
        - radius 1
        - UNK to replace all identifiers that appear less than 4 times
        - skip-gram and window size of 10
        - embeddings size 300
    '''
    featurizer = dc.feat.Mol2VecFingerprint()
    phi = featurizer.featurize(SMILES)

    #%% Model creation
    feature_extractor = net_(inp_dim=phi.shape[1],
                              out_dim=args.features,
                              n_layers=args.num_layers,
                              neurons_fc=args.neurons_fc,
                              hidden_activation=args.activation,
                              out_layer_activation=args.output_activation) 


    # Gaussian Regression model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    
    # load pre-trained models
    load_dir = 'pre-trained'
    # train data to load the model
    train_x = torch.tensor(torch.load(load_dir + "/train_x.pt")).to(device)
    train_y = torch.tensor(torch.load(load_dir + "/train_y.pt")).to(device)

    model = GPRegressionModel(train_x=train_x, 
                              train_y=train_y, 
                              likelihood=likelihood, 
                              feature_extractor=feature_extractor, 
                              args=args)
    
    

    
    model.load_state_dict(torch.load(load_dir + "/DKL_model.pt"))
    model.to(device)
    
    mu = torch.tensor(torch.load(load_dir + "/data_stats_mean.pt")).to(device)
    std = torch.tensor(torch.load(load_dir + "/data_stats_std.pt")).to(device)
          
    num_cores = multiprocessing.cpu_count()
    
    # target density fuel
    '''
    Screening SAF blends based on density [780, 790, 800]
    '''
    fuel_target = 780
    fuel = 	'SAF-{}'.format(fuel_target)
    alpha = [0.1, 0.5, 1.0]
    K = 1000
        
        
    # Save diretory
    results_dir = "results/" + fuel
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    
    # Optimization
    for idx in range(3):
        output_filename = results_dir + '/output_optimization_fuel_{}_lasso_{}_samples_{}.txt'.format(fuel,alpha[idx],K)
        print(output_filename)
    
        f = open(output_filename, "w")
        f.write(">>>>>>>>>>>>>>>>>>>>  Output File for Inverse fuel design with scipy minimize  <<<<<<<<<<<<<<<<\n\n")
        f.write(f"Rodolfo Freitas      {date_time}\n")
        f.write(f"GPU name: {torch.cuda.get_device_name(device=device)}\n")
        f.write(f"number of cpus: {num_cores}\n")
        f.write(f"Target fuel density: {fuel_target}\n")
        f.write("-------------------------------------------------------------------------------------------------------\n\n")
        f.close()
        
    
        
        time_start = default_timer()
        sol = Parallel(n_jobs=num_cores)(delayed(opt)(k, alpha[idx], fuel_target)for k in range(K))
        time_end = default_timer()
        
        mins, secs = divmod(ceil(time_end-time_start), 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)
        
        
        
        f = open(output_filename, "a")
        f.write("-------------------------------------------------------------------------------------------------------\n\n")
        f.write(f"Total Optimization time: {days:02d}-{hours:02d}:{mins:02d}:{secs:02d}\n")
        f.write("-------------------------------------------------------------------------------------------------------\n\n")
        f.close()
        
        
        # save
        torch.save(sol, results_dir+'/compositions_optimization_fuel_{}_lasso_{}_samples_{}.pt'.format(fuel,alpha[idx],K))
        
