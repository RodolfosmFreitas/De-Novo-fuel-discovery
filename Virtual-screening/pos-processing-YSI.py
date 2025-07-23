# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:18:34 2025

@author: Rodolfo Freitas
"""

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
from models import net_, GPRegressionModel2
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
    

def predict_YSI(test_x):
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Test points are regularly spaced along [0,1]
        Y_pred = likelihood(model(test_x))

    # Sampling from the posterior
    f_samples = Y_pred.sample(sample_shape=torch.Size([1000])).detach().cpu().numpy() * std_YSI + mu_YSI 
    
    mu_pred = f_samples.mean(0) 
    var_pred = Y_pred.var(0)
    sigma_pred = np.sqrt(var_pred)
    return mu_pred, sigma_pred

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
    
    #%% read fuel data [Name, SMILES, CN, METHOD]
    # read fuel data [Name, SMILES, CN, METHOD]
    data = pd.read_excel('palette/fuel_palette_SAF.xlsx').to_numpy() 
    fuel_list = data[:28,0] # np.hstack((data[:28,0], data[33:,0])) 
    SMILES = data[:28,1].tolist() #+ data[33:,1].tolist() 
    
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
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    # load pre-trained models
    load_dir = 'pre_trained'
    # train data to load the model
    train_x = torch.tensor(torch.load(load_dir + "/train_x_YSI.pt")).to(device)
    train_y = torch.tensor(torch.load(load_dir + "/train_y_YSI.pt")).to(device)

    model = GPRegressionModel2(train_x=train_x, 
                                 train_y=train_y, 
                                 likelihood=likelihood, 
                                 feature_extractor=feature_extractor, 
                                 args=args)
   
    model.load_state_dict(torch.load(load_dir + "/DKL_model_YSI.pt"))
    model.to(device)
    
    
    
    # statistics of the properties 
    mu_YSI = torch.tensor(torch.load(load_dir + "/data_YSI_stats_mean.pt")).to(device)
    std_YSI = torch.tensor(torch.load(load_dir + "/data_YSI_stats_std.pt")).to(device)
    
    
    #%% read the optimization results
    fuel_target = 780
    fuel = 	'SAF-{}'.format(fuel_target)
    load_dir = "results/" + fuel
    
    filename = load_dir + "/{}_Blends_DeNovo_Design.xlsx".format(fuel)
    data    = pd.read_excel(filename, index_col=0)
    fuel_name   = data.index
    data_   = data.to_numpy() 
    mixtures = np.array(data_[:,:28], dtype=float)
    
    stats_YSI = np.zeros((mixtures.shape[0],2))
    
    for idx in range(mixtures.shape[0]):
        # Mixing Operator
        Phi = torch.tensor(np.matmul(mixtures[idx,:][None,:], phi), dtype=torch.float).to(device)
        # call the surrogate model
        stats_YSI[idx] = np.hstack(predict_YSI(Phi))
        
    
    # create a dataframe 
    d = {'YSI (mean)': stats_YSI[:,0],
          'YSI (std)': stats_YSI[:,1]}
    
    df2 = pd.DataFrame(data=d, index=['{}'.format(f) for f in fuel_name])
    
    df = pd.concat([data, df2], axis=1)
    df.to_excel(load_dir + "/{}_Blends_DeNovo_Design.xlsx".format(fuel)) 
    
