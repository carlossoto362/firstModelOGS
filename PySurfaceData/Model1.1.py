#!/usr/bin/env python

"""
Functions for learning the constants for the the invertion problem.

As part of the National Institute of Oceanography, and Applied Geophysics, I'm working on an invertion problem. A detailed description can be found at
https://github.com/carlossoto362/firstModelOGS.
the invertion model contains the functions required to reed the satellite data and process it, in order to obtain the constituents: (chl-a, CDOM, NAP), 
using the first introduced model.

In addition, some of the constants are now learnable parameters, and there is a function that uses the storical data to learn the parameters. 
"""
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
import pandas as pd
import os
import scipy
from scipy import stats
import time
import multiprocessing as mp

#reading the initial value for the constants from two csv files, one with the constants dependent on lambda, and the other with
#the ones that do not depend on it.


def reed_constants(file1='cte_lambda.csv',file2='cst.csv'):
    """
    function that reads the constants stored in file1 and file2. 
    file1 has the constants that are dependent on lambda, is a csv with the columns
    lambda, absortion_w, scattering_w, backscattering_w, absortion_PH, scattering_PH, backscattering_PH.
    file2 has the constants that are independent of lambda, is a csv with the columns
    name,values.

    reed_constants(file1,file2) returns a dictionary with all the constants. To access the absortion_w for examplea, write 
    constant = reed_constants(file1,file2)['absortion_w']['412.5'].
    """

    cts_lambda = pd.read_csv(file1)
    constant = {}
    for key in cts_lambda.keys()[1:]:
        constant[key] = {}
        for i in range(len(cts_lambda['lambda'])):
            constant[key][str(cts_lambda['lambda'].iloc[i])] = cts_lambda[key].iloc[i]
        cts = pd.read_csv(file2)
        
    for i in range(len(cts['name'])):
        constant[cts['name'].iloc[i]] = cts['value'].iloc[i]
    return constant

constant = reed_constants(file1='cte_lambda.csv',file2='cst.csv')
lambdas = np.array([412.5,442.5,490,510,555]).astype(float)

linear_regression=stats.linregress(lambdas,[constant['scattering_PH'][str(lamb)] for lamb in lambdas])
linear_regression_slope = linear_regression.slope
linear_regression_intercept = linear_regression.intercept

def reed_data(data_path='./SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS',train=True):
    """
    function that reads the data stored in SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS/surface.yyy-mm-dd_12-00-00.txt
    reed_data() returns a pandas DataFrame with all the data available on SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS.
    Each file has to be a csv with the columns
    lambda,RRS,E_dir,E_dif,zenit,PAR.
    
    """
    data = os.listdir(data_path)
    data.sort()
    if train == True:
        data = data[:int(len(data)*0.9)]
    else:
        data = data[int(len(data)*0.9):]
        
    names = ['lambda','RRS','E_dir','E_dif','zenit','PAR']
    all_data = pd.DataFrame(columns=['date','lambda','RRS','E_dir','E_dif','zenit','PAR'])
    for d in data:
        date = datetime.strptime(d,'surface.%Y-%m-%d_12-00-00.txt')
        one_data = pd.read_csv(data_path + '/' + d ,sep=' ',names=names)
        one_data['date'] = [date]*6
        all_data = pd.concat([all_data,one_data],ignore_index=True)
        
    return all_data.sort_values(by='date')


def reed_initial_conditions(data,results_path='./results_first_run',lambdas=[510.0,412.5,442.5,490.0,555.0]):
    """
    reeds the results stored in results_path. This data is used as initial conditions for the learnings, to make
    the learning process for each day faster. The files are supposed to be csv files with columns
    ,510.0,412.5,442.5,490.0,555.0,chla,CDOM,NAP,loss

    This function is supposed to be used with a DataFrame of data and a Dataframe of constants.
    Please create the global variables,
    
    >>>data = reed_data(train=train)
    >>>data = data[data['lambda'].isin(lambdas)]
    >>>data = reed_initial_conditions(data,results_path = './results_first_run')
    
    data is a pandas DataFrame with the columns
    date,lambda,RRS,E_dir,E_dif,zenit,PAR.
    
    reed_result reads from all the files on the path, so make sure that the path has no other file than the results. Each result file has the data of one date, and has
    to be stored in a file named %Y-%m-%d.csv. Is meant to be used after storing the data from the save_results function. 
    
    """
    results_names = os.listdir(results_path)
    dates_results = [datetime.strptime(d,'%Y-%m-%d.csv') for d in results_names]
    results = data[data['date'].isin(dates_results)]
    results['RRS_MODEL']=np.empty(len(results))
    results['chla']=np.empty(len(results))
    results['NAP']=np.empty(len(results))
    results['CDOM']=np.empty(len(results))
    results['loss']=np.empty(len(results))
    dates_ = results['date'].unique()
    for i in range(len(dates_)):
        d = dates_[i]
        results_i = pd.read_csv(results_path + '/' +d.strftime('%Y-%m-%d.csv'))
        for lamb in lambdas:
            results.loc[(results['date']==d) & (results['lambda']==lamb),'RRS_MODEL'] = float(results_i[str(lamb)])

        results.loc[(results['date']==d),'chla'] = float(results_i['chla'])
        results.loc[(results['date']==d),'CDOM'] = float(results_i['CDOM'])
        results.loc[(results['date']==d),'NAP'] = float(results_i['NAP'])
        results.loc[(results['date']==d),'loss'] = float(results_i['loss'])

    return results

def read_kd_data(data,kd_data_path='messure_data/kd_data/BOUSSOLEFit_AntoineMethod_filtered_butterHighpass6days_min.csv',lambdas=[510.0,412.5,442.5,490.0,555.0]):
    """
    reeds the kd data. The files are supposed to be csv files with columns
    510.0,412.5,442.5,490.0,555.0,chla,NAP,CDOM,loss

    This function is supposed to be used with a DataFrame of data, and a Dataframe of constants.
    Please create the global variables,
    
    >>>data = reed_data(train=train)
    >>>data = data[data['lambda'].isin(lambdas)]
    >>>data = reed_initial_conditions(data,results_path = './results_first_run')
    >>>data = read_kd_data(data)
    
    """
    kd_data = pd.read_csv(kd_data_path)
    kd_data['date'] = [datetime.strptime(kd_data['date'].iloc[k],'%Y-%m-%d') for k in range(len(kd_data))]
    result = data.merge(kd_data,on=['date','lambda'],how='left')
    return result

def read_chla_data(data,chl_data_path='messure_data/buoy.DPFF.2003-09-06_2012-12-31_999.dat',lambdas=[510.0,412.5,442.5,490.0,555.0]):
    """
    reeds the chla data. The files are supposed to be csv files, including a column with YEAR, MONTH, DAY and chl. 

    This function is supposed to be used with a DataFrame of data and a Dataframe of constants.
    Please create the global variables,
    
    >>>data = reed_data(train=train)
    >>>data = data[data['lambda'].isin(lambdas)]
    >>>data = reed_initial_conditions(data,results_path = './results_first_run')
    >>>data = read_kd_data(data)
    >>>data = read_chla_data(data)
    
    """
    buoy_data=pd.read_csv(chl_data_path,sep='\t')
    #print(buoy_data)
    
    buoy_results = np.empty((len(data)))
    for i in range(len(data)):
        date = data['date'].iloc[i]
        buoy_results[i] = buoy_data[ (buoy_data['YEAR'] == date.year) & (buoy_data['MONTH'] == date.month) & (buoy_data['DAY'] == date.day) ]['chl'].mean()
        data['buoy_chla'] = buoy_results
    return data

def read_bbp_data(data,bbp_data_path='messure_data/boussole_multi_rrs_bbpw_T10_IES20_2006-2012.csv',lambdas=[510.0,412.5,442.5,490.0,555.0]):
    """
    reeds the bbp data. The files are supposed to be csv files, including a column with YEAR, MONTH, DAY and bbp_550,bbp_488 and bbp_442. 

    This function is supposed to be used with a DataFrame of data and a Dataframe of constants.
    Please create the global variables,
    
    >>>data = reed_data(train=train)
    >>>data = data[data['lambda'].isin(lambdas)]
    >>>data = reed_initial_conditions(data,results_path = './results_first_run')
    >>>data = read_kd_data(data)
    >>>data = read_chla_data(data)
    >>>data = read_bbp_data(data)
    
    """
    buoy_data = pd.read_csv(bbp_data_path,sep=';')
    data['buoy_bbp'] = np.ones(len(data))*np.nan
    for i in range(len(data)):
        if (data['lambda'].iloc[i] == 510) or (data['lambda'].iloc[i] == 412.5):
            pass
        elif data['lambda'].iloc[i] == 555:
            year = data['date'].iloc[i].year
            month = data['date'].iloc[i].month
            day = data['date'].iloc[i].day
            data['buoy_bbp'].iloc[i] = buoy_data[(buoy_data['YEAR'] == year) & (buoy_data['MONTH'] == month) & (buoy_data['DAY'] == day) ]['bbp_550'].mean()
        elif data['lambda'].iloc[i] == 490:
            year = data['date'].iloc[i].year
            month = data['date'].iloc[i].month
            day = data['date'].iloc[i].day
            data['buoy_bbp'].iloc[i] = buoy_data[(buoy_data['YEAR'] == year) & (buoy_data['MONTH'] == month) & (buoy_data['DAY'] == day) ]['bbp_488'].mean()
        elif data['lambda'].iloc[i] == 442.5:
            year = data['date'].iloc[i].year
            month = data['date'].iloc[i].month
            day = data['date'].iloc[i].day
            data['buoy_bbp'].iloc[i] = buoy_data[(buoy_data['YEAR'] == year) & (buoy_data['MONTH'] == month) & (buoy_data['DAY'] == day) ]['bbp_442'].mean()
    return data
        

################Functions for the absortion coefitient####################
def absortion_CDOM(lambda_,perturbation_factors,tensor = True):
    """
    Function that returns the mass-specific absorption coefficient of CDOM, function dependent of the wavelength lambda. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == False:
        return constant['dCDOM']*np.exp(-(constant['sCDOM'] * perturbation_factors['mul_factor_s_cdom'])*(lambda_ - 450.))
    else:
        return constant['dCDOM']*torch.exp(-(constant['sCDOM'] * perturbation_factors['mul_factor_s_cdom'])*(torch.tensor(lambda_) - 450.))

def absortion_NAP(lambda_,tensor = True):
    """
    Mass specific absorption coefficient of NAP.
    See Gallegos et al., 2011.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == False:
    	return constant['dNAP']*np.exp(-constant['sNAP']*(lambda_ - 440.))
    else:
    	return constant['dNAP']*torch.exp(-constant['sNAP']*(torch.tensor(lambda_) - 440.))

def absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=True):
    """
    Total absortion coeffitient.
    aW,λ (values used from Pope and Fry, 1997), aP H,λ (values averaged and interpolated from
    Alvarez et al., 2022).
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return constant['absortion_w'][str(lambda_)] + (constant['absortion_PH'][str(lambda_)] * perturbation_factors['mul_factor_a_ph'])*chla + \
        (absortion_CDOM(lambda_, perturbation_factors,tensor=tensor)* perturbation_factors['mul_factor_a_cdom'])*CDOM + absortion_NAP(lambda_,tensor=tensor)*NAP


##############Functions for the scattering coefitient########################
def Carbon(chla,PAR, perturbation_factors,tensor=True):
    """
    defined from the carbon to Chl-a ratio. 
    theta_o, sigma, beta, and theta_min constants (equation and values computed from Cloern et al., 1995), and PAR
    the Photosynthetically available radiation, obtained from the OASIM model, see Lazzari et al., 2021.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    nominator = chla
    beta =  constant['beta'] * perturbation_factors['mul_factor_beta']
    sigma = constant['sigma'] * perturbation_factors['mul_factor_sigma']
    exponent = -(PAR - beta)/sigma
    if tensor == False:
        denominator = (constant['Theta_o']* perturbation_factors['mul_factor_theta_o']) * ( np.exp(exponent)/(1+np.exp(exponent)) ) + \
        (constant['Theta_min'] * perturbation_factors['mul_factor_theta_min'])
    else:
        denominator = (constant['Theta_o']* perturbation_factors['mul_factor_theta_o']) * ( torch.exp(exponent)/(1+torch.exp(exponent)) ) + \
        (constant['Theta_min'] * perturbation_factors['mul_factor_theta_min'])
    return nominator/denominator

def scattering_ph(lambda_,perturbation_factors,tensor = True):
    """
    The scattering_ph is defined initially as a linear regression between the diferent scattering_ph for each lambda, and then, I
    change the slope and the intercept gradually. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    
    return (linear_regression_slope * perturbation_factors['mul_tangent_b_ph']) *\
        lambda_ + linear_regression_intercept * perturbation_factors['mul_intercept_b_ph']

def scattering_NAP(lambda_,tensor=True):
    """
    NAP mass-specific scattering coefficient.
    eNAP and fNAP constants (equation and values used from Gallegos et al., 2011)
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return constant['eNAP']*(550./lambda_)**constant['fNAP']

def scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    Total scattering coefficient.
    bW,λ (values interpolated from Smith and Baker, 1981,), bP H,λ (values used
    from Dutkiewicz et al., 2015)
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return constant['scattering_w'][str(lambda_)] + scattering_ph(lambda_,perturbation_factors,tensor=tensor) * Carbon(chla,PAR,perturbation_factors,tensor=tensor) + \
        scattering_NAP(lambda_,tensor=tensor) * NAP

#################Functions for the backscattering coefitient#############

def backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    Total backscattering coefficient.
     Gallegos et al., 2011.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return constant['backscattering_w'][str(lambda_)] + constant['backscattering_PH'][str(lambda_)] * \
        Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP



###############Functions for the end solution of the equations###########
#The final result is written in terms of these functions, see ...

def c_d(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True): 
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
    	return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor) + scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/torch.cos(torch.tensor(zenit)*3.1416/180)
    else:
    	return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor) + scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/np.cos(zenit*3.1416/180)

def F_d(lambda_,zenit,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
    	return (scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) - constant['rd'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/\
        torch.cos(torch.tensor(zenit)*3.1416/180.)
    else:
    	return (scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) - constant['rd'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/\
        np.cos(zenit*3.1416/180.)

def B_d(lambda_,zenit,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
    	return  constant['rd']*backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor)/torch.cos(torch.tensor(zenit)*3.1416/180) 
    else:
    	return  constant['rd']*backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor)/np.cos(zenit*3.1416/180)

def C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor) + constant['rs'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) )/\
        constant['vs']

def B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (constant['ru'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/constant['vu']

def B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (constant['rs'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/constant['vs']

def C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor) + constant['ru'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/\
        constant['vu']

def D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (0.5) * (C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + \
                    ((C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor))**2 -\
                     4. * B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) * B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) )**(0.5))

def x(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    denominator = (c_d(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) - C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)) * \
        (c_d(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)) +\
        B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) * B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor)
    nominator = -(C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + c_d(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)) *\
        F_d(lambda_,zenit,PAR,chla,NAP,perturbation_factors,tensor=tensor) -\
        B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) * B_d(lambda_,zenit,PAR,chla,NAP,perturbation_factors,tensor=tensor)

    return nominator/denominator

def y(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    denominator = (c_d(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) - C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)) * \
        (c_d(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)) +\
        B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) * B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor)
    nominator = (-B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) * F_d(lambda_,zenit,PAR,chla,NAP,perturbation_factors,tensor=tensor) ) +\
        (-C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + c_d(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)) *\
        B_d(lambda_,zenit,PAR,chla,NAP,perturbation_factors,tensor=tensor)

    return nominator/denominator

def C_plus(E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return E_dif_o - x(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) * E_dir_o

def r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor)/D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)

def k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) - C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)

def E_dir(z,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    This is the analytical solution of the bio-chemical model. (work not published.)
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor==False:
        return E_dir_o*np.exp(-z*c_d(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor))
    else:
        return E_dir_o*torch.exp(-z*c_d(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors))
        

def E_u(z,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor = True):
    """
    This is the analytical solution of the bio-chemical model. (work not published.)
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == False:
        return C_plus(E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) * r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)*\
        np.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)*z)+\
        y(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)
    else:
        return C_plus(E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors) * r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors)*\
        torch.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors)*z)+\
        y(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors)

def E_dif(z,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    This is the analytical solution of the bio-chemical model. (work not published.)
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    if tensor == False:
        return C_plus(E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) *\
        np.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)*z)+\
        x(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)
    else:
        return C_plus(E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors) *\
        torch.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors)*z)+\
        x(lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors)
        

def bbp(E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    Particulate backscattering at depht z
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == False:
        return constant['backscattering_PH'][str(lambda_)] * \
        Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP
    else:
        return constant['backscattering_PH'][str(lambda_)] * \
        Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP

def kd(z,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    Atenuation Factor
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenit is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor==False:
        return (z**-1)*np.log((E_dir_o + E_dif_o)/(E_dir(z,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) +\
                                                  E_dif(z,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)))
    else:
        return (z**-1)*torch.log((E_dir_o + E_dif_o)/(E_dir(z,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors) +\
                                                  E_dif(z,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors)))

##########################from the bio-optical model to RRS(Remote Sensing Reflectance)##############################
#defining Rrs
#Q=5.33*np.exp(-0.45*np.sin(np.pi/180.*(90.0-Zenith)))

def Q_rs(zenit,perturbation_factors,tensor=True):
    """
    Empirical result for the Radiance distribution function, 
    equation from Aas and Højerslev, 1999, 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor==True:
        return (5.33 * perturbation_factors['mul_factor_q_a'])*torch.exp(-(0.45 * perturbation_factors['mul_factor_q_b'])*torch.sin((3.1416/180.0)*(90.0-torch.tensor(zenit))))
    else:
        return  (5.33 * perturbation_factors['mul_factor_q_a'])*np.exp(-(0.45 * perturbation_factors['mul_factor_q_b'])*np.sin((3.1416/180.0)*(90.0-zenit)))

def Rrs_minus(Rrs,tensor=True):
    """
    Empirical solution for the effect of the interface Atmosphere-sea.
     Lee et al., 2002
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return Rrs/(constant['T']+constant['gammaQ']*Rrs)

def Rrs_plus(Rrs,tensor=True):
    """
    Empirical solution for the effect of the interface Atmosphere-sea.
     Lee et al., 2002
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return Rrs*constant['T']/(1-constant['gammaQ']*Rrs)

def Rrs_MODEL(E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor = True):
    """
    Remote Sensing Reflectance.
    Aas and Højerslev, 1999.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    Rrs = E_u(0,E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)  /  (   Q_rs(zenit,perturbation_factors,tensor=tensor)*(E_dir_o + E_dif_o)   )
    return Rrs_plus( Rrs ,tensor = tensor)




#################################Putting all together#################################################
class MODEL(nn.Module):
    """
    Bio-Optical model plus corrections, in order to have the Remote Sensing Reflectance, in terms of the invertion problem. 
    MODEL(x) returns a tensor, with each component being the Remote Sensing Reflectance for each given wavelength. 
    if the data has 5 rows, each with a different wavelength, RRS will return a vector with 5 components.  RRS has tree parameters, 
    self.chla is the chlorophil-a, self.NAP the Non Algal Particles, and self.CDOM the Colored Dissolved Organic Mather. 
    According to the invention problem, we want to estimate them by making these three parameters have two constraints,
    follow the equations of the bio-optical model, plus, making the RRS as close as possible to the value
    measured by the satellite.
    
    """
    def __init__(self):
        super().__init__()
        self.chla = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.NAP = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.CDOM = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)

    def forward(self,x_data,perturbation_factors):
        """
        x_data: pandas dataframe with columns [E_dif,E_dir,lambda,zenit,PAR].
        """
        Rrs = torch.empty(len(x_data),)
        for i in range(len(x_data)):
            Rrs[i,] = Rrs_MODEL(x_data['E_dif'].iloc[i],x_data['E_dir'].iloc[i],x_data['lambda'].iloc[i],\
            x_data['zenit'].iloc[i],x_data['PAR'].iloc[i],self.chla,self.NAP,self.CDOM,perturbation_factors)
        return Rrs


    
def train_loop(data_i,model,loss_fn,optimizer,N,perturbation_factors):
    """
    The train loop evaluates the Remote Sensing Reflectance RRS for each wavelength>>>pred=model(data_i), evaluates the loss function
    >>>loss=loss_fn(pred,y), force the value of the parameters (chla,NAP,CDOM) to be positive, evaluates the gradient of RRS with respect
    to the parameters, >>>loss.backward(), modifies the value of the parameters according to the optimizer criterium, >>>optimizer.step(),
    sets the gradient of RRS to cero, and prints the loss for a given number of iterations. This procedure is performed N times or untyl a treshold is ashieved. 
    After N iterations, it returns two lists with the evolution of the loss function and the last evaluation of the model. 
    """
    size = len(data_i)
    data_i = data_i.loc[:,data_i.columns!='date'].astype(float)
    ls_val=[]
    ls_count=[]

    criterium = 1
    criterium_2 = 0
    i=0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #file_ = open('results_imprubment.txt','w')
    while (((criterium >1e-12 ) & (i<N)) or criterium_2 < 100):
        y = data_i['RRS'].to_numpy()
        y = torch.tensor(y).float()
        pred = model(data_i,perturbation_factors)
        loss = loss_fn(pred,y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for p in model.parameters():
            p.data.clamp_(0)
        
        if i % 1 == 0:
            ls_val.append(loss.item())
            ls_count.append(i)
            #print(i,list(model.parameters()))
            if i != 0:
                criterium = ls_val[-2] - ls_val[-1]
        if criterium <=0:
            criterium_2+=1
        i+=1
        
        scheduler.step(loss)
    #file_.close()
    return ls_val,ls_count,pred

def run_invertion(x_data_tensor_T,perturbation_factors,cores=40):
	"""
	Designed to make multiple inversion problems in paralel manner. Because of conflicts with pytorch, I end up dont making it on parallen manually.

	It runs the invertion problem for N diferent days.

	x_data_tensor_T is a 26*N torch.tensor, with each column representing the quantities:
	col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26
        RRS555,RRS510,RRS490,RRS442,RRS412,Edif555,Edif510,Edif490,Edif442,Edif412,Edir555,Edir510,
        Edir490,Edir442,Edir412,lambda555,lambda510,lambda490,lambda442,lambda412,zenith,PAR,chla,nap,cdom,date

	perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
	of some of the constants of the bio quimical model.

	The output is a 9*N torch.tensor, with columns representing
	col1,col2,col3,col4,col5,col6,col7,col8,col9
        chla,kd555,kd510,kd490,kd442,kd412,bbp555,bbp490,bbp442
	"""
	#time_one_batch = time.time()
	x_data_size = x_data_tensor_T.size()[0]
	x_result = torch.empty(x_data_size,9)
	#pool = mp.Pool(cores)
	input = []
	for ind in range(x_data_size):
		input.append([ind,x_data_tensor_T[ind],perturbation_factors])
	results_ = []
	for input_i in input:
		results_.append(run_one_invertion(input_i))
	#results_ = pool.map(run_one_invertion,input)
	#print('One minibatch done..., time used: ' + str(time.time() - time_one_batch))
	for result_ in results_:
		x_result[result_[0]] = result_[1]
	
	#pool.close()
	return x_result

def run_one_invertion(input):
        """
	Runs the invertion problem for the bio-optical model, using the loss_function MSELoss(), and the optimizer torch.optim.Adam.

	This function is supposed to be used with a DataFrame of data and a Dataframe of constants.

	input is an iterable, with three elements, the first is an index, in order preserved the ordering of the elements, in case is needed. 
	the second is x_data_tensor, descrived in the function run_invertion, and the last one is the perturbation_factors, also described in
	run_inversion.

        """
    
        x_data_tensor = input[1]
        x_index = input[0]
        perturbation_factors = input[2]
        #timeInit=time.time()
        x_data = pd.DataFrame()
        dates_ = []
        lambdas = []
        RRS=[]
        Edif=[]
        Edir=[]
        zenith=[]
        PAR=[]
        chla=[]
        nap=[]
        cdom=[]
        for lam in range(5):
            lambdas.append(float(x_data_tensor[15+lam]))
            RRS.append(float(x_data_tensor[lam]))
            Edif.append(float(x_data_tensor[5+lam]))
            Edir.append(float(x_data_tensor[10+lam]))
            zenith.append(float(x_data_tensor[20]))
            PAR.append(float(x_data_tensor[21]))
            chla.append(float(x_data_tensor[22]))
            nap.append(float(x_data_tensor[23]))
            cdom.append(float(x_data_tensor[24]))
            dates_.append(datetime(year=2000,month=1,day=1) + timedelta(float(x_data_tensor[25])))
        x_data['RRS'] = RRS
        x_data['E_dif'] = Edif
        x_data['E_dir'] = Edir
        x_data['zenit'] = zenith
        x_data['PAR'] = PAR
        x_data['chla'] = chla
        x_data['NAP'] = nap
        x_data['CDOM'] = cdom
        x_data['lambda'] = lambdas
        x_data['date'] = dates_
        x_data = x_data.sort_values(by=['lambda'],ascending=False)

        N=4000
    
        model = MODEL().to("cpu")
        state_dict = model.state_dict()
        state_dict['chla'] = torch.ones((1,1), dtype=torch.float32)*x_data['chla'].iloc[0]#initial conditions equal to the first run (with the base parameters)
        state_dict['NAP'] = torch.ones((1,1), dtype=torch.float32)*x_data['NAP'].iloc[0]
        state_dict['CDOM'] = torch.ones((1,1), dtype=torch.float32)*x_data['CDOM'].iloc[0]
        model.load_state_dict(state_dict)
    
        learning_rate = 5e-3 #this is to use gradient descent. 
        loss_function = nn.MSELoss() #MSE, the same used by Paolo.
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        #optimizer = torch.optim.LBFGS(model.parameters(),lr=learning_rate) #this requires the closure function, but is much slower.

        ls_val,ls_count,pred = train_loop(x_data,model,loss_function,optimizer,N,perturbation_factors)

        for par in model.parameters():
            par.require_grad=False

        chla = list(model.parameters())[0]
        nap = list(model.parameters())[1]
        cdom = list(model.parameters())[2]
        x_result_i = torch.empty(9)
        x_result_i[0] = list(model.parameters())[0]

        for i in range(len(x_data)):
            if x_data['lambda'].iloc[i]==555:
                x_result_i[6] = bbp(x_data['E_dif'].iloc[i],x_data['E_dir'].iloc[i],x_data['lambda'].iloc[i],x_data['zenit'].iloc[i],\
                                    x_data['PAR'].iloc[i],chla,nap,cdom,perturbation_factors)
            elif x_data['lambda'].iloc[i]==490:
                x_result_i[7] = bbp(x_data['E_dif'].iloc[i],x_data['E_dir'].iloc[i],x_data['lambda'].iloc[i],x_data['zenit'].iloc[i],\
                                    x_data['PAR'].iloc[i],chla,nap,cdom,perturbation_factors)
            elif x_data['lambda'].iloc[i]==442.5:
                x_result_i[8] = bbp(x_data['E_dif'].iloc[i],x_data['E_dir'].iloc[i],x_data['lambda'].iloc[i],x_data['zenit'].iloc[i],\
                                    x_data['PAR'].iloc[i],chla,nap,cdom,perturbation_factors)
            

            x_result_i[i+1] =  kd(9.,x_data['E_dif'].iloc[i],x_data['E_dir'].iloc[i],x_data['lambda'].iloc[i],x_data['zenit'].iloc[i],\
                            x_data['PAR'].iloc[i],chla,nap,cdom,perturbation_factors)
        #print('time used to save '+ x_data['date'].iloc[0].strftime('%Y-%m-%d') + ', '+str(time.time() - timeInit)+' seconds')
        return x_index,x_result_i



class customTensorData():
    """
	Custom class to load the data and transform it into tensors that are meant to be used with the DataLoader function of pytorch. 
    """
    def __init__(self, initial_conditions_path,data_path,train=True, transform=None, target_transform=None):

        timeInitI = time.time()
        initial_perturbation_factors = {
            'mul_factor_a_ph':1,
            'mul_tangent_b_ph':1,
            'mul_intercept_b_ph':1,
            'mul_factor_a_cdom':1,
            'mul_factor_s_cdom':1,
            'mul_factor_q_a':1,
            'mul_factor_q_b':1,
            'mul_factor_theta_min':1,
            'mul_factor_theta_o':1,
            'mul_factor_beta':1,
            'mul_factor_sigma':1,
            'mul_factor_backscattering_ph':1,
            'mul_factor_backscattering_nap':1,
        }
        
        data = reed_data(train=train)
        data = data[data['lambda'].isin(lambdas)]
        data = reed_initial_conditions(data,results_path = './results_first_run')
        data = read_kd_data(data)
        data = read_chla_data(data)
        data = read_bbp_data(data)
        data['kd'] = [kd(9,data['E_dif'].iloc[i],data['E_dir'].iloc[i],data['lambda'].iloc[i],\
                     data['zenit'].iloc[i],data['PAR'].iloc[i],data['chla'].iloc[i],data['NAP'].iloc[i],data['CDOM'].iloc[i],\
                     initial_perturbation_factors,tensor=False) for i in range(len(data))]
        
    
        data['bbp'] = [bbp(data['E_dif'].iloc[i],data['E_dir'].iloc[i],data['lambda'].iloc[i],\
                     data['zenit'].iloc[i],data['PAR'].iloc[i],data['chla'].iloc[i],data['NAP'].iloc[i],data['CDOM'].iloc[i],\
                     initial_perturbation_factors,tensor=False) for i in range(len(data))]
        data = data.sort_values(by=['date','lambda'])
        labels = pd.DataFrame()
        labels['chla_d'] = data['buoy_chla'][data['lambda']==555].to_numpy()
        labels['kd_d_555'] = data['kd_filtered_min'][data['lambda']==555].to_numpy()
        labels['kd_d_510'] = data['kd_filtered_min'][data['lambda']==510].to_numpy()
        labels['kd_d_490'] = data['kd_filtered_min'][data['lambda']==490].to_numpy()
        labels['kd_d_442'] = data['kd_filtered_min'][data['lambda']==442.5].to_numpy()
        labels['kd_d_412'] = data['kd_filtered_min'][data['lambda']==412.5].to_numpy()
        labels['bbp_d_555'] = data['buoy_bbp'][data['lambda']==555].to_numpy()
        labels['bbp_d_490'] = data['buoy_bbp'][data['lambda']==490].to_numpy()
        labels['bbp_d_442'] = data['buoy_bbp'][data['lambda']==442.5].to_numpy()

        images = pd.DataFrame()

        images['RRS_555'] =  data['RRS'][data['lambda']==555].to_numpy()
        images['RRS_510'] = data['RRS'][data['lambda']==510].to_numpy()
        images['RRS_490'] = data['RRS'][data['lambda']==490].to_numpy()
        images['RRS_442'] = data['RRS'][data['lambda']==442.5].to_numpy()
        images['RRS_412'] = data['RRS'][data['lambda']==412.5].to_numpy()
        
        images['E_dif_555'] =  data['E_dif'][data['lambda']==555].to_numpy()
        images['E_dif_510'] = data['E_dif'][data['lambda']==510].to_numpy()
        images['E_dif_490'] = data['E_dif'][data['lambda']==490].to_numpy()
        images['E_dif_442'] = data['E_dif'][data['lambda']==442.5].to_numpy()
        images['E_dif_412'] = data['E_dif'][data['lambda']==412.5].to_numpy()

        images['E_dir_555'] =  data['E_dir'][data['lambda']==555].to_numpy()
        images['E_dir_510'] = data['E_dir'][data['lambda']==510].to_numpy()
        images['E_dir_490'] = data['E_dir'][data['lambda']==490].to_numpy()
        images['E_dir_442'] = data['E_dir'][data['lambda']==442.5].to_numpy()
        images['E_dir_412'] = data['E_dir'][data['lambda']==412.5].to_numpy()

        images['lambda_555'] =  data['lambda'][data['lambda']==555].to_numpy()
        images['lambda_510'] = data['lambda'][data['lambda']==510].to_numpy()
        images['lambda_490'] = data['lambda'][data['lambda']==490].to_numpy()
        images['lambda_442'] = data['lambda'][data['lambda']==442.5].to_numpy()
        images['lambda_412'] = data['lambda'][data['lambda']==412.5].to_numpy()

        images['zenith'] = data['zenit'][data['lambda']==555].to_numpy()
        images['PAR'] = data['PAR'][data['lambda']==555].to_numpy()

        images['chla'] = data['chla'][data['lambda']==555].to_numpy()
        images['NAP'] = data['NAP'][data['lambda']==555].to_numpy()
        images['CDOM'] = data['CDOM'][data['lambda']==555].to_numpy()
        

        images['date'] = [(d - datetime(year=2000,month=1,day=1)).days for d in data['date'][data['lambda']==555].iloc[:] ]
        
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.images = images
        self.dates = data['date'][data['lambda']==555]

        if train ==True:
            print('Time spended in reading the train data: ' + str(time.time() - timeInitI))
            print('shape of train data: ',self.images.shape)
        else:
            print('Time spended in reading the test data: ' + str(time.time() - timeInitI))
            print('shape of test data: ',self.images.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Now that im transforming the data in tensors, I'm going to loos the names, so the order is important
        col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17
        RRS555,RRS510,RRS490,RRS442,RRS412,Edif555,Edif510,Edif490,Edif442,Edif412,Edir555,Edir510,
        Edir490,Edir442,Edir412,lambda555,lambda510,lambda490,lambda442,lambda412,zenith,PAR,chla,nap,cdom

        and for the labels,
        col1,col2,col3,col4,col5,col6,col7,col8,col9
        chla,kd555,kd510,kd490,kd442,kd412,bbp555,bbp490,bbp442
        """
        image = torch.tensor(self.images.iloc[idx].values)
        label = torch.tensor(self.labels.iloc[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class Parameter_Estimator(nn.Module):
    """
	Model that attempts to learn the perturbation factors. 
    """
    def __init__(self):
        super().__init__()
        self.mul_factor_a_ph = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.mul_tangent_b_ph = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.mul_intercept_b_ph = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.mul_factor_a_cdom = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.mul_factor_s_cdom = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.mul_factor_q_a = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.mul_factor_q_b = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.mul_factor_theta_min = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.mul_factor_theta_o = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.mul_factor_sigma = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.mul_factor_backscattering_ph = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.mul_factor_backscattering_nap = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.mul_factor_beta = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)

        self.perturbation_factors = {
            'mul_factor_a_ph':self.mul_factor_a_ph,
            'mul_tangent_b_ph':self.mul_tangent_b_ph,
            'mul_intercept_b_ph':self.mul_intercept_b_ph,
            'mul_factor_a_cdom':self.mul_factor_a_cdom,
            'mul_factor_s_cdom':self.mul_factor_s_cdom,
            'mul_factor_q_a':self.mul_factor_q_a,
            'mul_factor_q_b':self.mul_factor_q_b,
            'mul_factor_theta_min':self.mul_factor_theta_min,
            'mul_factor_theta_o':self.mul_factor_theta_o,
            'mul_factor_beta':self.mul_factor_beta,
            'mul_factor_sigma':self.mul_factor_sigma,
            'mul_factor_backscattering_ph':self.mul_factor_backscattering_ph,
            'mul_factor_backscattering_nap':self.mul_factor_backscattering_nap,
    }

    def forward(self,image,cores=40):
        """
        x_data: pandas dataframe with columns [E_dif,E_dir,lambda,zenit,PAR].
        """

        return run_invertion(image,self.perturbation_factors,cores=cores)

def custom_LSELoss(input,output):
	"""
	My data has some nan on it, so this function returns the least square error loss function, taking into consideration the nan elements.
	"""
	custom_array = (output-input)**2
	sum_ = 0
	k_ = 0
	for element in custom_array:
		sum2_ = 0
		k2_ = 0
		for element2 in element:

			if math.isnan(element2):
				pass
			else:
				sum2_ += element2
				k2_ += 1
		if k2_ == 0: 
			pass
		else:
			sum_ += sum2_/k2_
			k_ +=1
	if k_ == 0 :
		return input.mean()*torch.rand(1)
	else:
		return sum_/k_

class WeightClipper(object):
    """
	Object that clamps the parameters between 0 and 2. 
    """
    def __init__(self):
        pass

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0.001,1.999)
            module.weight.data = w

def train_loop_parameters(dataloader, model_par, loss_fn_par, optimizer_par,cores=40):
    """
	train loop for the learning of the perturbation factors. It has an scheduler in order to increase the velocity of convergence, and a cliper, to set 
	a constrains in the parameters. 
    """
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    num_batch = 0
    training_parameters_file = open('training_parameters.csv','w')
    
    scheduler_par = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_par, 'min')
    for batch, (X, y) in enumerate(dataloader):
        one_batch_time = time.time()
        # Compute prediction and loss
        pred = model_par(X,cores=cores)
        loss = loss_fn_par(pred, y)
        num_batch+=1
        print('batch: ',num_batch,'time used: ',time.time() - one_batch_time,'loss: ',loss.item())
        print(list(model_par.parameters()),file = training_parameters_file)
        # Backpropagation
        loss.backward()
        optimizer_par.step()
        optimizer_par.zero_grad()

        cliper = WeightClipper()
        model_par.apply(cliper)

        scheduler_par.step(loss)


    training_parameters_file.close()

def test_loop_parameters(dataloader, model_par, loss_fn_par,cores=40):
    """
	test loop, that evaluates the model with the dataloader. 
    """
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model_par.eval()
    
    num_batches = len(dataloader)
    test_loss = 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    for param in model_par.parameters():
        param.requires_grad = False

    #pool = mp.Pool(cores)
    #results_ = pool.map(test_loss_add,[(X_,model_par,loss_fn_par) for X_ in dataloader])
    #test_loss = [result_[0] for result_ in results_ ]
    #pool.close()
    #test_loss = np.mean(test_loss)
    num_batch = 0
    for X, y in dataloader:
        one_batch_time = time.time()
        pred = model_par(X,cores=cores)
        #print(pred)
        #chla,kd555,kd510,kd490,kd442,kd412,bbp555,bbp490,bbp442
        #print(y)
        test_loss += loss_fn_par(pred, y)
        num_batch+=1
        print('batch: ',num_batch,'time used: ',time.time() - one_batch_time,'loss: ',test_loss/num_batch)
    test_loss = test_loss/num_batch
    for param in model_par.parameters():
        param.requires_grad = True
    print(f"Avg loss: {test_loss:>8f} \n")




if __name__ == "__main__":

    time_zero=time.time()
    device = (
	    "cuda"
	    if torch.cuda.is_available()
	    else "cpu"
    )
    print(f"Using {device} device")
    initial_conditions_path = '/Users/carlos/Documents/surface_data_analisis/results_first_run'
    data_path = '/Users/carlos/Documents/surface_data_analisis/SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS'
    train_data = customTensorData(initial_conditions_path,data_path)
    test_data = customTensorData(initial_conditions_path,data_path,train=False)

    learning_rate = 1
    batch_size = 40
    epochs = 10

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,num_workers=40)
    test_dataloader = DataLoader(test_data, batch_size=5, shuffle=False,num_workers=40)

    model_par = Parameter_Estimator()
    #loss_fn_par = nn.MSELoss()
    loss_fn_par = custom_LSELoss

    optimizer_par = torch.optim.SGD(model_par.parameters(), lr=learning_rate)

    #next_iteration_image,next_iteration_label = next(iter(train_dataloader))
    for t in range(epochs):
        print("Epoch: ",t)
        train_loop_parameters(train_dataloader,model_par,loss_fn_par,optimizer_par)
        test_loop_parameters(test_dataloader, model_par, loss_fn_par,cores=40)
    print("Done")

    #for i, (batch_x, batch_y) in enumerate(test_dataloader):
    #    print(f"Batch {i}: input shape {batch_x.shape}, label shape {batch_y.shape}")
    initial_perturbation_factors = {
            'mul_factor_a_ph':1,
            'mul_tangent_b_ph':1,
            'mul_intercept_b_ph':1,
            'mul_factor_a_cdom':1,
            'mul_factor_s_cdom':1,
            'mul_factor_q_a':1,
            'mul_factor_q_b':1,
            'mul_factor_theta_min':1,
            'mul_factor_theta_o':1,
            'mul_factor_beta':1,
            'mul_factor_sigma':1,
            'mul_factor_backscattering_ph':1,
            'mul_factor_backscattering_nap':1,
    }    

    #result = run_invertion(next_iteration_image,initial_perturbation_factors)
    print('total time for: '+ str(time.time() - time_zero))
    
