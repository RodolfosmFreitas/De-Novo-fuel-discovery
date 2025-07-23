# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:39:43 2024

@author: exy029
"""

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import gpytorch
from matplotlib import pyplot as plt
import time
import argparse
import numpy as np
from models import net_, GPRegressionModel
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import deepchem as dc
import seaborn as sns


# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Publication-style formatting
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "lines.linewidth": 1.5,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# Train
parser = argparse.ArgumentParser(description='Deep Kernel learning + Molecular representation')

parser.add_argument('--data-dir', type=str, default="../data", help='data directory')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learnign rate')
parser.add_argument('--n-iterations', type=int, default=100, help='number of iterations to train (default: 1000)')
parser.add_argument('--train-size', type=float, default=0.80, help='amount of the data used to train')
parser.add_argument('--log-interval', type=int, default=10, help='how many epochs to wait before logging training status')

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

# Check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('------------ Arguments -------------')
print("Torch device:{}-{}".format(device,torch.cuda.get_device_name(0)))
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


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


#%% Preparing the data for model (pure compounds)

# read fuel data [Name, SMILES, CN, METHOD]
data            = pd.read_excel('{}/data_density_LHV.xlsx'.format(args.data_dir)).to_numpy() 
fuel_list_pure  = data[:,0]
SMILES          = data[:,1].tolist()

# load the dataset [density (kg/m3), LHV (MJ/kg)]
Y_pure = data[:,2:].astype(np.float64)

X_pure = featurizer.featurize(SMILES)

# Split the data in train and test dataset 
X_train_pure, X_test_pure, Y_train_pure, Y_test_pure, fuel_train_pure, fuel_test_pure = train_test_split(X_pure,
                                                                                                         Y_pure,
                                                                                                         fuel_list_pure,
                                                                                                         train_size=args.train_size, 
                                                                                                         shuffle=True)

# Data Augmentation: data + noise * N(0,1)
noise = 0.05 * Y_train_pure
samples = 5
X_train_p = np.repeat(X_train_pure, samples, axis=0)
Y_train_p = np.repeat(Y_train_pure, samples, axis=0) + np.repeat(noise,samples, axis=0) * np.random.randn(noise.shape[0]*samples,2)

#%% Preparing the data for model (blends)

# read fuel data [Name, SMILES, CN, METHOD]
data_blends    = pd.read_excel('{}/data_density_LHV_TF_blends.xlsx'.format(args.data_dir))
fuel_list_blends = data_blends['Fuels'].to_numpy()
SMILES_blends = list(data_blends.keys()[1:-4])

# load the properties
Y_blends = data_blends.to_numpy()[:,-3:-1].astype(np.float64)

# molecular representation of fuel blends
feat_blends = featurizer.featurize(SMILES_blends)

# Mixing Operator
W = data_blends[SMILES_blends].to_numpy()
    
X_blends = np.matmul(W,feat_blends)

# Split the data in train and test dataset 
X_train_blends, X_test_blends, Y_train_blends, Y_test_blends, fuel_train_blends, fuel_test_blends = train_test_split(X_blends,
                                                                                                                     Y_blends,
                                                                                                                     fuel_list_blends,
                                                                                                                     train_size=args.train_size, 
                                                                                                                     shuffle=True)

# Data Augmentation: data + noise * N(0,1)
noise = 0.05 * Y_train_blends
X_train_b = np.repeat(X_train_blends, samples, axis=0)
Y_train_b = np.repeat(Y_train_blends, samples, axis=0) + np.repeat(noise,samples, axis=0) * np.random.randn(noise.shape[0]*samples,2)
 

#%% Concatenate pure compounds and blends
X_train = np.concatenate((X_train_p, X_train_b), axis=0)
Y_train = np.concatenate((Y_train_p, Y_train_b), axis=0)


# pre-processing the data (Scaling)
mu = np.concatenate((Y_train_pure, Y_train_blends), axis=0).mean(0)
std = np.concatenate((Y_train_pure, Y_train_blends), axis=0).std(0)
Y_train = ((Y_train - mu)/ std) 

# transform to tensor
X_train = torch.Tensor(X_train).to(device)
Y_train = torch.Tensor(Y_train).to(device)


#%% MODEL
feature_extractor = net_(inp_dim=X_train.shape[1],
                          out_dim=args.features,
                          n_layers=args.num_layers,
                          neurons_fc=args.neurons_fc,
                          hidden_activation=args.activation,
                          out_layer_activation=args.output_activation) 

print("Feature Extractor Descriptors: number of parameters {}".format(feature_extractor.num_parameters()))

# Gaussian Regression model
#likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)

model = GPRegressionModel(train_x=X_train, 
                          train_y=Y_train, 
                          likelihood=likelihood, 
                          feature_extractor=feature_extractor, 
                          args=args)

print("Gaussian Process:", model)

# 
feature_extractor.to(device)
likelihood.to(device)
model.to(device)

# Save diretory
model_dir = "Models/DKL_FeatureExtractor_{}x{}_kernel_{}".\
    format(args.num_layers, args.neurons_fc,args.kernel)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.save(mu, model_dir + "/data_stats_mean.pt")
torch.save(std, model_dir + "/data_stats_std.pt")
torch.save(X_train, model_dir + "/train_x.pt")
torch.save(Y_train, model_dir + "/train_y.pt")
#%% ================================ Training ===================================#

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},], 
    lr=args.lr, weight_decay = 1e-5)


# "Loss" for GPs - the marginal log likelihood
#mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# How to schedule the learning rate
scheduler = lr_scheduler.MultiStepLR(optimizer, 
                                      milestones=[0.1 * args.n_iterations + idx * (0.1 * args.n_iterations) for idx in range(int(args.n_iterations/(0.1 * args.n_iterations))-1)], 
                                      gamma=0.75)

print("Start training ...")

tic = time.time()
start_time = tic
train_loss = []

for iter in range(1,args.n_iterations+1):
    model.train()
    likelihood.train()
    
    # Zero gradients from previous iteration
    optimizer.zero_grad() 
    
    # Output from model
    y_pred = model(X_train)
    
    # Compute the negative marginal log likelihood loss 
    loss = -mll(y_pred, Y_train)
    
    # Backward Step 
    loss.backward()
    optimizer.step()
    
    # Save loss
    train_loss.append(loss.item())
    
    
        
    # Print
    if iter % args.log_interval == 0:
         
        elapsed = time.time() - start_time
        print('Iteration: %d, Time: %.2f, Train loss: %.3f' % 
              (iter, elapsed, loss.item()))
        start_time = time.time()
    
    scheduler.step()
    

tic2 = time.time()
print("Done training {} iteratiosn in {} seconds"
      .format(args.n_iterations, tic2 - start_time))

# save the model
torch.save(model.state_dict(), model_dir + "/DKL_model.pt")

# Save the training details
torch.save(train_loss, model_dir + "/loss_train.pt")

plt.figure(figsize=(6.5,3.5))
plt.plot(train_loss, 'b-')
plt.ylabel(r'- log p($y|\Phi,\theta, w$)')
plt.xlabel(r'Number of Iterations')
plt.savefig(model_dir+'/marginal_log_likelihood.jpg')
plt.show()

#%% Make predictions
'''
    model(test_x) returns the model posterior distribution p(f* | x*, X, y)'
    likelihood(model(test_x)) gives us the posterior predictive distribution p(y* | x*, X, y) 
    which is the probability distribution over the predicted output value
'''


def predict(test_x):
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Test points are regularly spaced along [0,1]
        return likelihood(model(test_x))
    
# Get into evaluation (predictive posterior) mode and predict

# train
X_train = np.concatenate((X_train_pure, X_train_blends), axis=0)
Y_train = np.concatenate((Y_train_pure, Y_train_blends), axis=0)
X_train = torch.Tensor(X_train).to(device) 
Y_pred_train = predict(X_train) 
    
# Sampling from the posterior
f_samples_train = Y_pred_train.sample(sample_shape=torch.Size([1000])).detach().cpu().numpy() * std + mu
    
# Compute mean and variance of the prediction as function of x
mu_pred_train = f_samples_train.mean(0)    
var_pred_train = f_samples_train.var(0)   

# "Diagnostics: How do you know if the fit is good?"

mape_train = mean_absolute_percentage_error(Y_train, mu_pred_train, multioutput='raw_values')
mae_train = mean_absolute_error(Y_train, mu_pred_train, multioutput='raw_values') / np.mean(Y_train, axis=0)
rmse_train = root_mean_squared_error(Y_train, mu_pred_train, multioutput='raw_values')
r2_train = r2_score(Y_train, mu_pred_train, multioutput='raw_values')

print("Default selection Train MAE:",mae_train)
print("Default selection R2-score:",r2_train)


# test 
X_test = np.concatenate((X_test_pure, X_test_blends), axis=0)
Y_test = np.concatenate((Y_test_pure, Y_test_blends), axis=0)
X_test = torch.Tensor(X_test).to(device)
Y_pred_test = predict(X_test)

# Sampling from the posterior
f_samples_test = Y_pred_test.sample(sample_shape=torch.Size([1000])).detach().cpu().numpy() * std + mu
    
# Compute mean and variance of the prediction as function of x
mu_pred_test = f_samples_test.mean(0)    
var_pred_test = f_samples_test.var(0)  

# "Diagnostics: How do you know if the fit is good?"

mape_test = mean_absolute_percentage_error(Y_test, mu_pred_test, multioutput='raw_values')
mae_test = mean_absolute_error(Y_test, mu_pred_test, multioutput='raw_values') / np.mean(Y_test, axis=0)
rmse_test = root_mean_squared_error(Y_test, mu_pred_test, multioutput='raw_values')
r2_test = r2_score(Y_test, mu_pred_test, multioutput='raw_values')

print("Default selection Train MAE:",mae_test)
print("Default selection R2-score:",r2_test)


# creata a dataframe
d = {'Train [Density, LHV]': [mape_train, mae_train, rmse_train, r2_train], #
      'Test [Density, LHV]': [mape_test, mae_test, rmse_test, r2_test]}

metrics = pd.DataFrame(data=d, index=['MAPE', 'nMAE', 'RMSE', 'R2'])

metrics.to_excel(model_dir + '/metrics.xlsx')

# Density 
min_data  = np.minimum(np.minimum(np.amin(Y_train[:,0]),np.amin(mu_pred_train[:,0])), np.minimum(np.amin(Y_test[:,0]), np.amin(mu_pred_test[:,0])))
max_data = np.maximum(np.maximum(np.amax(Y_train[:,0]),np.amax(mu_pred_train[:,0])), np.maximum(np.amax(Y_test[:,0]),np.amax((mu_pred_test[:,0]))))

colors = ['#e7298a', '#1b9e77']

plt.figure(figsize=(6.5,3.5))

plt.scatter(np.reshape(mu_pred_train[:,0],-1),np.reshape(Y_train[:,0],-1), 
                  s=20 ,marker='s', color = colors[0],  alpha=0.5, label=r'Train')
acc_train = (f'R$^2$ = {r2_train[0]:.3f}\n'
        f'nMAE={mae_train[0]:.3f}')

bbox = dict(boxstyle='round', fc=colors[0], ec=colors[0], alpha=0.15)
plt.text(1000., 400, acc_train, bbox=bbox, horizontalalignment='right')


plt.scatter(np.reshape(mu_pred_test[:,0],-1),np.reshape(Y_test[:,0],-1), 
                  s=30 ,marker='o', color = colors[1], alpha=0.75, label=r'Test')
acc_test = (f'R$^2$ = {r2_test[0]:.3f}\n'
        f'nMAE={mae_test[0]:.3f}')
bbox = dict(boxstyle='round', fc=colors[1], ec=colors[1], alpha=0.15)
plt.text(1000., 520, acc_test, bbox=bbox, horizontalalignment='right')

xlim = plt.xlim(min_data, max_data)
ylim = plt.ylim(min_data, max_data)
plt.plot(xlim,ylim,'k--')
ticks = np.linspace(400, 1200, 6)
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlabel(r'Predicted Density [kg/m$^3$]')
plt.ylabel(r'Measured Density [kg/m$^3$]')
plt.legend(loc='best', frameon=False, prop={'weight': 'extra bold'})
plt.savefig(model_dir + '/parity_density.pdf', bbox_inches='tight')
plt.show()

# LHV 
min_data  = np.minimum(np.minimum(np.amin(Y_train[:,1]),np.amin(mu_pred_train[:,1])), np.minimum(np.amin(Y_test[:,1]), np.amin(mu_pred_test[:,1])))
max_data = np.maximum(np.maximum(np.amax(Y_train[:,1]),np.amax(mu_pred_train[:,1])), np.maximum(np.amax(Y_test[:,1]),np.amax((mu_pred_test[:,1]))))

plt.figure(figsize=(6.5,3.5))

plt.scatter(np.reshape(mu_pred_train[:,1],-1),np.reshape(Y_train[:,1],-1), 
                  s=20 ,marker='s', color = colors[0],  alpha=0.5, label=r'Train')
acc_train = (f'R$^2$ = {r2_train[1]:.3f}\n'
        f'nMAE={mae_train[1]:.3f}')

bbox = dict(boxstyle='round', fc=colors[0], ec=colors[0], alpha=0.15)
plt.text(45., 20, acc_train, bbox=bbox, horizontalalignment='right')


plt.scatter(np.reshape(mu_pred_test[:,1],-1),np.reshape(Y_test[:,1],-1), 
                  s=30 ,marker='o', color = colors[1], alpha=0.75, label=r'Test')
acc_test = (f'R$^2$ = {r2_test[1]:.3f}\n'
        f'nMAE={mae_test[1]:.3f}')
bbox = dict(boxstyle='round', fc=colors[1], ec=colors[1], alpha=0.15)
plt.text(45., 25, acc_test, bbox=bbox, horizontalalignment='right')

xlim = plt.xlim(min_data, max_data)
ylim = plt.ylim(min_data, max_data)
plt.plot(xlim,ylim,'k--')
ticks = np.linspace(20, 50, 6)
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlabel(r'Predicted LHV [MJ/kg]')
plt.ylabel(r'Measured LHV [MJ/kg]')
plt.legend(loc='best', frameon=False, prop={'weight': 'extra bold'})
plt.savefig(model_dir + '/parity_LHV.pdf', bbox_inches='tight')
plt.show()

e_train = (mu_pred_train - Y_train) / np.sqrt(var_pred_train)
e_test = (mu_pred_test - Y_test) / np.sqrt(var_pred_test)

zs = np.linspace(-3, 3, 100)

# kernel density estimation


plt.figure(figsize=(6.5,3.5))
sns.kdeplot(e_train[:,0], fill=True, color=colors[0], linewidth=2, alpha=0.25, label=r'Train')
sns.kdeplot(e_test[:,0], fill=True, color=colors[1], linewidth=2, alpha=0.25, label=r'Test')
plt.axvline(x=0, color='black', linestyle='--')
plt.ylabel(r'Distribution')
plt.xlabel(r'Std. error')
plt.legend(loc='best', frameon=False, prop={'weight': 'extra bold'})
plt.xlim([-4,4])
plt.savefig(model_dir + '/PDF_error_density.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6.5,3.5))
sns.kdeplot(e_train[:,1], fill=True, color=colors[0], linewidth=2, alpha=0.25, label=r'Train')
sns.kdeplot(e_test[:,1], fill=True, color=colors[1], linewidth=2, alpha=0.25, label=r'Test')
plt.axvline(x=0, color='black', linestyle='--')
plt.ylabel(r'Distribution')
plt.xlabel(r'Std. error')
plt.legend(loc='best', frameon=False, prop={'weight': 'extra bold'})
plt.xlim([-1,1])
plt.savefig(model_dir + '/PDF_error_LHV.pdf', bbox_inches='tight')
plt.show()


#%% Pure and blends
 
Y_pred_test_pure = predict(torch.Tensor(X_test_pure).to(device))
Y_pred_test_blends = predict(torch.Tensor(X_test_blends).to(device))

# Compute mean and variance of the prediction as function of x
mu_pred_test_pure = Y_pred_test_pure.mean.detach().cpu().numpy() * std + mu      
mu_pred_test_blends = Y_pred_test_blends.mean.detach().cpu().numpy() * std + mu     

# Density
min_data  = np.minimum(np.minimum(np.amin(Y_train[:,0]),np.amin(mu_pred_train[:,0])), np.minimum(np.amin(Y_test[:,0]), np.amin(mu_pred_test[:,0])))
max_data = np.maximum(np.maximum(np.amax(Y_train[:,0]),np.amax(mu_pred_train[:,0])), np.maximum(np.amax(Y_test[:,0]),np.amax((mu_pred_test[:,0]))))

plt.figure(figsize=(6.5,3.5))
plt.scatter(np.reshape(mu_pred_test_pure[:,0],-1),np.reshape(Y_test_pure[:,0],-1), 
                  s=30, marker='X', color = '#1f77b4',  alpha=0.75, label=r'Pure ({})'.format(Y_test_pure.shape[0]))
plt.scatter(np.reshape(mu_pred_test_blends[:,0],-1),np.reshape(Y_test_blends[:,0],-1), 
                  s=30 ,marker='p', color = 'magenta', alpha=0.75, label=r'Blends ({})'.format(Y_test_blends.shape[0]))
xlim = plt.xlim(min_data, max_data)
ylim = plt.ylim(min_data, max_data)
plt.plot(xlim,ylim,'k--')
plt.xlabel(r'Predicted Density [kg/m$^3$]')
plt.ylabel(r'Measured Density [kg/m$^3$]')
ticks = np.linspace(400, 1200, 6)
plt.xticks(ticks)
plt.yticks(ticks)
plt.legend(loc='best', frameon=False, prop={'weight': 'extra bold'})
plt.savefig(model_dir + '/parity_density_test_pure_blends.pdf', bbox_inches='tight')
plt.show()


#LHV
min_data  = np.minimum(np.minimum(np.amin(Y_train[:,1]),np.amin(mu_pred_train[:,1])), np.minimum(np.amin(Y_test[:,1]), np.amin(mu_pred_test[:,1])))
max_data = np.maximum(np.maximum(np.amax(Y_train[:,1]),np.amax(mu_pred_train[:,1])), np.maximum(np.amax(Y_test[:,1]),np.amax((mu_pred_test[:,1]))))

plt.figure(figsize=(6.5,3.5))
plt.scatter(np.reshape(mu_pred_test_pure[:,1],-1),np.reshape(Y_test_pure[:,1],-1), 
                  s=30, marker='X', color = '#1f77b4',  alpha=0.75, label=r'Pure ({})'.format(Y_test_pure.shape[0]))
plt.scatter(np.reshape(mu_pred_test_blends[:,1],-1),np.reshape(Y_test_blends[:,1],-1), 
                  s=30 ,marker='p', color = 'magenta', alpha=0.75, label=r'Blends ({})'.format(Y_test_blends.shape[0]))
xlim = plt.xlim(min_data, max_data)
ylim = plt.ylim(min_data, max_data)
plt.plot(xlim,ylim,'k--')
ticks = np.linspace(20, 50, 6)
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlabel(r'Predicted LHV [MJ/kg]')
plt.ylabel(r'Measured LHV [MJ/kg]')
plt.legend(loc='best', frameon=False, prop={'weight': 'extra bold'})
plt.savefig(model_dir + '/parity_LHV_test_pure_blends.pdf', bbox_inches='tight')
plt.show()


# Subplot setup
fig, axes = plt.subplots(1, 3, figsize=(7.205, 2.4), sharex=False)

axes[0].scatter(np.reshape(mu_pred_train[:,0],-1),np.reshape(Y_train[:,0],-1), 
                  s=20 ,marker='s', color = colors[0],  alpha=0.5, label=r'Train')
acc_train = (f'R$^2$ = {r2_train[0]:.3f}\n'
        f'nMAE={mae_train[0]:.3f}')
bbox = dict(boxstyle='round', fc=colors[0], ec=colors[0], alpha=0.15)
axes[0].text(1300., 380, acc_train, bbox=bbox, horizontalalignment='right')
#
axes[0].scatter(np.reshape(mu_pred_test[:,0],-1),np.reshape(Y_test[:,0],-1), 
                  s=30 ,marker='o', color = colors[1], alpha=0.75, label=r'Test')
acc_test = (f'R$^2$ = {r2_test[0]:.3f}\n'
        f'nMAE={mae_test[0]:.3f}')
bbox = dict(boxstyle='round', fc=colors[1], ec=colors[1], alpha=0.15)
axes[0].text(1300., 580, acc_test, bbox=bbox, horizontalalignment='right')
#
min_data  = np.minimum(np.minimum(np.amin(Y_train[:,0]),np.amin(mu_pred_train[:,0])), np.minimum(np.amin(Y_test[:,0]), np.amin(mu_pred_test[:,0])))
max_data = np.maximum(np.maximum(np.amax(Y_train[:,0]),np.amax(mu_pred_train[:,0])), np.maximum(np.amax(Y_test[:,0]),np.amax((mu_pred_test[:,0]))))
xlim = plt.xlim(min_data, max_data)
ylim = plt.ylim(min_data, max_data)
axes[0].plot(xlim,ylim,'k--')
axes[0].set_xlabel(r'Predicted Density [kg/m$^3$]')
axes[0].set_ylabel(r'Measured Density [kg/m$^3$]')
ticks = np.linspace(400, 1200, 3)
axes[0].set_xticks(ticks)
axes[0].set_yticks(ticks)
axes[0].legend(loc='upper left', frameon=False, prop={'weight': 'extra bold'})
sns.kdeplot(e_train[:,0], ax=axes[1], fill=True, color=colors[0], linewidth=2, alpha=0.25, label=r'Train')
sns.kdeplot(e_test[:,0], ax=axes[1], fill=True, color=colors[1], linewidth=2, alpha=0.25, label=r'Test')
axes[1].axvline(x=0, color='black', linestyle='--')
axes[1].set_ylabel(r'Distribution')
axes[1].set_xlabel(r'Std. error')
axes[1].legend(loc='upper left', frameon=False, prop={'weight': 'extra bold'})
axes[1].set_xlim([-4,4])

axes[2].scatter(np.reshape(mu_pred_test_pure[:,0],-1),np.reshape(Y_test_pure[:,0],-1), 
                  s=30, marker='X', color = '#1f77b4',  alpha=0.75, label=r'Pure ({})'.format(Y_test_pure.shape[0]))
axes[2].scatter(np.reshape(mu_pred_test_blends[:,0],-1),np.reshape(Y_test_blends[:,0],-1), 
                  s=30 ,marker='p', color = 'magenta', alpha=0.75, label=r'Blends ({})'.format(Y_test_blends.shape[0]))
xlim = plt.xlim(min_data, max_data)
ylim = plt.ylim(min_data, max_data)
axes[2].plot(xlim,ylim,'k--')
axes[2].set_xlabel(r'Predicted Density [kg/m$^3$]')
axes[2].set_ylabel(r'Measured Density [kg/m$^3$]')
axes[2].set_xticks(ticks)
axes[2].set_yticks(ticks)
axes[2].legend(loc='upper left', frameon=False, prop={'weight': 'extra bold'})
fig.tight_layout(rect=[0, 0.01, 1, 0.98])  # leave space for suptitle
# Save figure
plt.savefig(model_dir + "/Density_panel.pdf", bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(7.205, 2.4), sharex=False)

axes[0].scatter(np.reshape(mu_pred_train[:,1],-1),np.reshape(Y_train[:,1],-1), 
                  s=20 ,marker='s', color = colors[0],  alpha=0.5, label=r'Train')
acc_train = (f'R$^2$ = {r2_train[1]:.3f}\n'
        f'nMAE={mae_train[1]:.3f}')
bbox = dict(boxstyle='round', fc=colors[0], ec=colors[0], alpha=0.15)
axes[0].text(50, 17., acc_train, bbox=bbox, horizontalalignment='right')
#
axes[0].scatter(np.reshape(mu_pred_test[:,1],-1),np.reshape(Y_test[:,1],-1), 
                  s=30 ,marker='o', color = colors[1], alpha=0.75, label=r'Test')
acc_test = (f'R$^2$ = {r2_test[1]:.3f}\n'
        f'nMAE={mae_test[1]:.3f}')
bbox = dict(boxstyle='round', fc=colors[1], ec=colors[1], alpha=0.15)
axes[0].text(50, 25, acc_test, bbox=bbox, horizontalalignment='right')
#
min_data  = np.minimum(np.minimum(np.amin(Y_train[:,1]),np.amin(mu_pred_train[:,1])), np.minimum(np.amin(Y_test[:,1]), np.amin(mu_pred_test[:,1])))
max_data = np.maximum(np.maximum(np.amax(Y_train[:,1]),np.amax(mu_pred_train[:,1])), np.maximum(np.amax(Y_test[:,1]),np.amax((mu_pred_test[:,1]))))
xlim = plt.xlim(min_data, max_data)
ylim = plt.ylim(min_data, max_data)
axes[0].plot(xlim,ylim,'k--')
axes[0].set_xlabel(r'Predicted LHV [MJ/kg]')
axes[0].set_ylabel(r'Measured LHV [MJ/kg]')
ticks = np.linspace(20, 50, 3)
axes[0].set_xticks(ticks)
axes[0].set_yticks(ticks)
axes[0].legend(loc='best', frameon=False, prop={'weight': 'extra bold'})

sns.kdeplot(e_train[:,1], ax=axes[1], fill=True, color=colors[0], linewidth=2, alpha=0.25, label=r'Train')
sns.kdeplot(e_test[:,1], ax=axes[1], fill=True, color=colors[1], linewidth=2, alpha=0.25, label=r'Test')
axes[1].axvline(x=0, color='black', linestyle='--')
axes[1].set_ylabel(r'Distribution')
axes[1].set_xlabel(r'Std. error')
axes[1].legend(loc='upper left', frameon=False, prop={'weight': 'extra bold'})
axes[1].set_xlim([-1,1])

axes[2].scatter(np.reshape(mu_pred_test_pure[:,1],-1),np.reshape(Y_test_pure[:,1],-1), 
                  s=30, marker='X', color = '#1f77b4',  alpha=0.75, label=r'Pure ({})'.format(Y_test_pure.shape[0]))
axes[2].scatter(np.reshape(mu_pred_test_blends[:,1],-1),np.reshape(Y_test_blends[:,1],-1), 
                  s=30 ,marker='p', color = 'magenta', alpha=0.75, label=r'Blends ({})'.format(Y_test_blends.shape[0]))
xlim = plt.xlim(min_data, max_data)
ylim = plt.ylim(min_data, max_data)
axes[2].plot(xlim,ylim,'k--')
axes[2].set_xlabel(r'Predicted LHV [MJ/kg]')
axes[2].set_ylabel(r'Measured LHV [MJ/kg]')
axes[2].set_xticks(ticks)
axes[2].set_yticks(ticks)
axes[2].legend(loc='upper left', frameon=False, prop={'weight': 'extra bold'})

fig.tight_layout(rect=[0, 0.01, 1, 0.98])  # leave space for suptitle
# Save figure
plt.savefig(model_dir + "/LHV_panel.pdf", bbox_inches='tight')
plt.show()



# Compute the errors and the uncertainty
errors_train = np.zeros((Y_train.shape[0],2))
uncertainty_train = np.zeros((Y_train.shape[0],2))

for idx in range(Y_train.shape[0]):
    s_pred_sample = mu_pred_train[idx]
    s_pred_uncertainty = np.sqrt(var_pred_train[idx])
    s_test_sample = Y_train[idx]

    errors_train[idx] = np.square(s_pred_sample - s_test_sample) / np.square(s_test_sample) 
    uncertainty_train[idx] = np.square(s_pred_uncertainty) / np.square(s_test_sample)


# Compute the errors and the uncertainty
errors = np.zeros((Y_test.shape[0],2))
uncertainty = np.zeros((Y_test.shape[0],2))

for idx in range(Y_test.shape[0]):
    s_pred_sample = mu_pred_test[idx]
    s_pred_uncertainty = np.sqrt(var_pred_test[idx])
    s_test_sample = Y_test[idx]

    errors[idx] = np.abs(s_pred_sample - s_test_sample) / np.abs(s_test_sample)
    uncertainty[idx] = np.abs(s_pred_uncertainty) / np.abs(s_test_sample)


errors_idx_density = errors[:,0] < .05
errors_density = errors[errors_idx_density,0]
uncertainty_density = uncertainty[errors_idx_density,0]

errors_idx_LHV = errors[:,1] < .05
errors_LHV = errors[errors_idx_LHV,1]
uncertainty_LHV = uncertainty[errors_idx_LHV,1]


plt.figure(figsize=(3.5, 3.5),dpi=300)
plt.scatter(errors_density,uncertainty_density, 
                  s=20 ,marker='o', color = colors[1], alpha=0.75, label=r'Test')
plt.ylabel(r'Uncertainty - Density')
plt.xlabel(r'Error - Density')
plt.savefig(model_dir +'/Error_uncertainty_density.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(3.5, 3.5),dpi=300)
plt.scatter(errors_LHV,uncertainty_LHV, 
                  s=20 ,marker='o', color = colors[1], alpha=0.75)
plt.ylabel(r'Uncertainty - LHV')
plt.xlabel(r'Error - LHV')
plt.savefig(model_dir +'/Error_uncertainty_LHV.pdf', bbox_inches='tight')
plt.show()
