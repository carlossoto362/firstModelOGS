import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from datetime import datetime, timedelta
import pandas as pd
import os

#reading the constants from two csv files, one with the constants dependent on lambda, and the other with
#the ones that not depend on it.

def reed_constants(file1,file2):
    """
    constants stored in cte_lambda.csv and cst.csv
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

constant = reed_constants('cte_lambda.csv','cst.csv')

def reed_data():
    """
    the data is in SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS/surface.yyy-mm-dd_12-00-00.txt
    """
    data = os.listdir('SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS')
    names = ['lambda','RRS','E_dir','E_dif','zenit','PAR']
    all_data = pd.DataFrame(columns=['date','lambda','RRS','E_dir','E_dif','zenit','PAR'])
    for d in data:
        date = datetime.strptime(d,'surface.%Y-%m-%d_12-00-00.txt')
        one_data = pd.read_csv('SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS/' + d ,sep=' ',names=names)
        one_data['date'] = [date]*6
        all_data = all_data.append(one_data,ignore_index=True)
        
    return all_data.sort_values(by='date')


################Functions for the absortion coefitient####################
def absortion_CDOM(lambda_):
    """
    Mass specific absortion coefitient of CDOM.
    """
    return constant['dCDOM']*np.exp(-constant['sCDOM']*(lambda_ - 450))

def absortion_NAP(lambda_):
    """
    Mass specific absortion coefitient of NAP.
    """
    return constant['dNAP']*np.exp(-constant['sNAP']*(lambda_ - 440))

def absortion(lambda_,chla,CDOM,NAP):
    """
    Total absortion coeffitient.
    """
    return constant['absortion_w'][str(lambda_)] + constant['absortion_PH'][str(lambda_)]*chla + \
        absortion_CDOM(lambda_)*CDOM + absortion_NAP(lambda_)*NAP


##############Functions for the scattering coefitient########################
def Carbon(chla,PAR):
    """
    defined from the carbon to Chl-a ratio: 
    """
    return (constant['Theta_o'] *  (  np.exp(-(PAR - constant['beta'])/constant['sigma'])/\
                                  (1+np.exp(-(PAR - constant['beta'])/constant['sigma'])  )) + constant['Theta_min'])*chla

def scattering_NAP(lambda_):
    """
    NAP mass-specific scattering coeffitient.
    """
    return constant['eNAP']*(550/lambda_)**constant['fNAP']

def scattering(lambda_,PAR,chla,NAP):
    """
    Total scattering coeffitient.
    """
    return constant['scattering_w'][str(lambda_)] + constant['scattering_PH'][str(lambda_)] * Carbon(chla,PAR) + \
        scattering_NAP(lambda_) * NAP

#################Functions for the backscattering coefitient#############

def backscattering(lambda_,PAR,chla,NAP):
    """
    Total backscattering coeffitient.
    """
    return constant['backscattering_w'][str(lambda_)] + constant['backscattering_PH'][str(lambda_)] * \
        Carbon(chla,PAR) + 0.005 * scattering_NAP(lambda_) * NAP



###############Functions for the end solution of the equations###########

def c_d(lambda_,zenit,PAR,chla,NAP,CDOM):
    return (absortion(lambda_,chla,CDOM,NAP) + scattering(lambda_,PAR,chla,NAP))/np.cos(90-zenit)

def F_d(lambda_,zenit,PAR,chla,NAP):
    """
    The final result is writen in terms of this functions.
    """
    return (scattering(lambda_,PAR,chla,NAP) - constant['rd'] * backscattering(lambda_,PAR,chla,NAP))/\
        np.cos(90-zenit)

def B_d(lambda_,zenit,PAR,chla,NAP):
    return  constant['rd']*backscattering(lambda_,PAR,chla,NAP)/np.cos(90-zenit) 

def C_s(lambda_,PAR,chla,NAP,CDOM):
    return (absortion(lambda_,chla,CDOM,NAP) + constant['rs'] * backscattering(lambda_,PAR,chla,NAP) )/\
        constant['vs']

def B_u(lambda_,PAR,chla,NAP):
    return (constant['ru'] * backscattering(lambda_,PAR,chla,NAP))/constant['vu']

def B_s(lambda_,PAR,chla,NAP):
    return (constant['rs'] * backscattering(lambda_,PAR,chla,NAP))/constant['vs']

def C_u(lambda_,PAR,chla,NAP,CDOM):
    return (absortion(lambda_,chla,CDOM,NAP) + constant['ru'] * backscattering(lambda_,PAR,chla,NAP))/\
        constant['vu']

def D(lambda_,PAR,chla,NAP,CDOM):
    return (1/2) * (C_s(lambda_,PAR,chla,NAP,CDOM) + C_u(lambda_,PAR,chla,NAP,CDOM) + \
                    ((C_s(lambda_,PAR,chla,NAP,CDOM) + C_u(lambda_,PAR,chla,NAP,CDOM))**2 -\
                     4 * B_s(lambda_,PAR,chla,NAP) * B_u(lambda_,PAR,chla,NAP) )**(0.5))

def x(lambda_,zenit,PAR,chla,NAP,CDOM):
    denominator = (c_d(lambda_,zenit,PAR,chla,NAP,CDOM) - C_s(lambda_,PAR,chla,NAP,CDOM)) * \
        (c_d(lambda_,zenit,PAR,chla,NAP,CDOM) + C_u(lambda_,PAR,chla,NAP,CDOM)) +\
        B_s(lambda_,PAR,chla,NAP) * B_u(lambda_,PAR,chla,NAP)
    nominator = -(C_u(lambda_,PAR,chla,NAP,CDOM) + c_d(lambda_,zenit,PAR,chla,NAP,CDOM)) * F_d(lambda_,zenit,PAR,chla,NAP) -\
        B_u(lambda_,PAR,chla,NAP) * B_d(lambda_,zenit,PAR,chla,NAP)

    return nominator/denominator

def y(lambda_,zenit,PAR,chla,NAP,CDOM):
    denominator = (c_d(lambda_,zenit,PAR,chla,NAP,CDOM) - C_s(lambda_,PAR,chla,NAP,CDOM)) * \
        (c_d(lambda_,zenit,PAR,chla,NAP,CDOM) + C_u(lambda_,PAR,chla,NAP,CDOM)) +\
        B_s(lambda_,PAR,chla,NAP) * B_u(lambda_,PAR,chla,NAP)
    nominator = (-B_s(lambda_,PAR,chla,NAP) * F_d(lambda_,zenit,PAR,chla,NAP) ) +\
        (-C_s(lambda_,PAR,chla,NAP,CDOM) + c_d(lambda_,zenit,PAR,chla,NAP,CDOM)) * B_d(lambda_,zenit,PAR,chla,NAP)

    return nominator/denominator

def C_plus(E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM):
    return E_dif_o - x(lambda_,zenit,PAR,chla,NAP,CDOM) * E_dir_o

def r_plus(lambda_,PAR,chla,NAP,CDOM):
    return B_s(lambda_,PAR,chla,NAP)/D(lambda_,PAR,chla,NAP,CDOM)

def E_u_o(E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM):
    return C_plus(E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM) * r_plus(lambda_,PAR,chla,NAP,CDOM) +\
        y(lambda_,zenit,PAR,chla,NAP,CDOM) * E_dir_o

#defining Rrs
#Q=5.33*np.exp(-0.45*np.sin(np.pi/180.*(90.0-Zenith)))

def Q_rs(zenit):
    return 5.33*np.exp(-0.45*np.sin((np.pi/180)*(90.0-zenit)))

def Rrs_minus(Rrs):
    return Rrs/(constant['T']+constant['gammaQ']*Rrs)

def Rrs_plus(Rrs):
    return Rrs*constant['T']/(1-constant['gammaQ']*Rrs)

def Rrs_MODEL(E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM):
    return Rrs_plus( E_u_o(E_dif_o,E_dir_o,lambda_,zenit,PAR,chla,NAP,CDOM)/(Q_rs(zenit)*(E_dir_o - E_dif_o) ))




#deffin the model, hyperparameters, loss function and optimizer
class MODEL(nn.Module):

    def __init__(self):
        super().__init__()
        self.chla = nn.Parameter(torch.ones((1,1), dtype=torch.float32)*60, requires_grad=True)
        self.NAP = nn.Parameter(torch.ones((1,1), dtype=torch.float32)*50, requires_grad=True)
        self.CDOM = nn.Parameter(torch.ones((1,1), dtype=torch.float32)*30, requires_grad=True)

    def forward(self,x):
        """
        x: pandas dataframe with columns [E_dif,E_dir,lambda,zenit,PAR].
        """
        Rrs = torch.empty(len(x),)
        for i in range(len(x)):
            Rrs[i,] = Rrs_MODEL(x['E_dif'].iloc[i],x['E_dir'].iloc[i],x['lambda'].iloc[i],x['zenit'].iloc[i],\
                            x['PAR'].iloc[i],self.chla,self.NAP,self.CDOM)
        return Rrs


    
def train_loop(data_i,model,loss_fn,optimizer,N):
    
    size = len(data_i)
    data_i = data_i.loc[:,data_i.columns!='date'].astype(float)
    ls_val=[]
    ls_count=[]
    

    for i in range(N):
        y = data_i['RRS'].to_numpy()
        y = torch.tensor(y).float()
        pred = model(data_i)
        loss = loss_fn(pred,y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        for p in model.parameters():
            p.data.clamp_(0)
        if i % 1000 == 0:
            ls_val.append(loss.item())
            ls_count.append(i)
            print(ls_val[-1],ls_count[-1])
    return ls_val,ls_count,pred


#plt.plot(ls_count,ls_val)
#plt.show()


    
    
