# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 04:59:38 2024

@author: Rodolfo Freitas
"""

import torch
import gpytorch
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import deepchem as dc
from models import net_, GPRegressionModel
from scipy import stats
import seaborn as sns
import os
import argparse


# Train
parser = argparse.ArgumentParser(description='Deep Kernel learning + Molecular representation')

parser.add_argument('--data-dir', type=str, default="../data", help='data directory')

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



def predict(test_x):
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Test points are regularly spaced along [0,1]
        Y_pred = likelihood(model(test_x))
    
    mu_pred = Y_pred.mean * std_ + mu_
    var_pred = Y_pred.variance * std_ + mu_
    sigma_pred = torch.sqrt(var_pred)
    
    return mu_pred.detach().cpu().numpy(), sigma_pred.detach().cpu().numpy()
    


def harmonic_mixing_rule(w, X):
    '''
    Model-Based Formulation of Biofuel Blends by Simultaneous Product and Pathway Design
    Manuel Dahmen and Wolfgang Marquardt
    Energy & Fuels 2017 31 (4), 4096-4121
    DOI: 10.1021/acs.energyfuels.7b00118
    '''
    return np.dot(w, X)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #%% 
    data = pd.read_excel('{}/fuel_palette_SAF.xlsx'.format(args.data_dir)).to_numpy()
    fuel_list   =  data[:28,0]
    SMILES      = data[:28,1].tolist() 
    nC          = data[:28,3] 
    MWs         = data[:28,5] 
    epsilon     = data[:28,6]   
    HC          = data[:28,7]
    Hs          = data[:28,8] 
    FPs         = data[:28,9] 
    STs         = data[:28,10] 
    classes     = data[:28,-1].tolist() 
    
    
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
    
    #%% Import ML models for predict CN and YSI
    feature_extractor = net_(inp_dim=phi.shape[1],
                              out_dim=args.features,
                              n_layers=args.num_layers,
                              neurons_fc=args.neurons_fc,
                              hidden_activation=args.activation,
                              out_layer_activation=args.output_activation) 

    print("Feature Extractor Descriptors: number of parameters {}".format(feature_extractor.num_parameters()))

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
    
    # statistics of the properties 
    mu_ = torch.tensor(torch.load(load_dir + "/data_stats_mean.pt")).to(device)
    std_ = torch.tensor(torch.load(load_dir + "/data_stats_std.pt")).to(device)
    
    #%% read the optimization results
    
    '''
    SAFs
    Density = [780, 790, 800]
    '''
    fuel_target = 780
    fuel = 	'SAF-{}'.format(fuel_target)
    alpha = [0.1, 0.5, 1.0] 
    K = 1000
    
    save_dir = "results/" + fuel
    
    
    filename = save_dir + '/output_optimization_fuel_{}_lasso_{}_samples_{}.txt'.format(fuel,alpha,K)
    
    
    sol = []
    for idx in range(len(alpha)):  
        sol += torch.load(save_dir+'/compositions_optimization_fuel_{}_lasso_{}_samples_{}.pt'.format(fuel,alpha[idx],K))

    loss = []
    # matrix of compositions
    mixtures = []
    n_components = []
    
    # tolerance for possible solutions
    threshold = 1
    for idx in range(len(sol)):
        x_, _ = sol[idx]
        # remove molecules that contribute less the 1% to the mixture and rescale
        x_ = np.where(x_ > 1e-2, x_, 0.0)
        x_ = x_ / x_.sum()
        
        # density
        molecularW = harmonic_mixing_rule(x_, MWs)
        Flash_P = harmonic_mixing_rule(x_, FPs)
        if (Flash_P >= 38):

            # Aviation fuel composition constraints: aromatics (8–25%)
            if (x_[21:28].sum() >= 0.08 and x_[21:28].sum() <= 0.25): 
                Phi = torch.tensor(np.matmul(x_[None,:], phi), dtype=torch.float).to(device)
                mu, sigma = predict(Phi)
                
                loss_ = np.mean(np.square(mu[0][0] - fuel_target))
                
                if loss_ < threshold:
                    print(idx)
                    mixtures.append(x_)
                    loss.append(loss_)
                    n_components.append(np.count_nonzero(x_))
    
    mixtures = np.array(mixtures)
    

    #%% save results 
    stats_rho = np.zeros((mixtures.shape[0],2))
    stats_LHV = np.zeros((mixtures.shape[0],2))
    molecular_weights = np.zeros(mixtures.shape[0])
    dieletric = np.zeros(mixtures.shape[0])
    flash_points = np.zeros(mixtures.shape[0])
    surface_tensions = np.zeros(mixtures.shape[0])
    H_C = np.zeros(mixtures.shape[0])
        
    for idx in range(mixtures.shape[0]):
        # Mixing Operator
        Phi = torch.tensor(np.matmul(mixtures[idx,:][None,:], phi), dtype=torch.float).to(device)
        # call the surrogate model
        mu, sigma = predict(Phi)
        stats_rho[idx] = np.hstack((mu[0][0], sigma[0][0]))
        stats_LHV[idx] = np.hstack((mu[0][1], sigma[0][1]))
        
        # compute other physical properties
        molecular_weights[idx] = harmonic_mixing_rule(mixtures[idx], MWs)
        flash_points[idx] = harmonic_mixing_rule(mixtures[idx], FPs)
        dieletric[idx] = harmonic_mixing_rule(mixtures[idx], epsilon)
        surface_tensions[idx] = harmonic_mixing_rule(mixtures[idx], STs)
        H_C[idx] = harmonic_mixing_rule(mixtures[idx], HC)
            
    
    
    df1 = pd.DataFrame(data=mixtures, 
                        index=['{}-{}'.format(fuel, b) for b in range(mixtures.shape[0])], 
                        columns=fuel_list.tolist())
    
    # create a dataframe 
    d = {'MW (g/mol)': molecular_weights, 
          'density (mean) [kg/m^3]': stats_rho[:,0],
          'density (std) [kg/m^3]': stats_rho[:,1],
          'dieletric constant': dieletric,
          'H/C ratio': H_C,
          'Flash point [C]': flash_points,
          'LHV (mean) [MJ/kg]': stats_LHV[:,0],
          'LHV (std) [MJ/kg]': stats_LHV[:,1],
          'surface tension (mN/m)': surface_tensions}
    
    df2 = pd.DataFrame(data=d, index=['{}-{}'.format(fuel, b) for b in range(mixtures.shape[0])])
    df = pd.concat([df1, df2], axis=1)
    df.to_excel(save_dir + "/{}_Blends_DeNovo_Design.xlsx".format(fuel)) 
    
               