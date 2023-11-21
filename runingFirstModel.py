#!/usr/bin/env python

"""
running the inversion model. 

As part of the National Institute of Oceanography, and Applied Geophysics, I'm working on an inversion problem. A detailed description can be found at 
https://github.com/carlossoto362/firstModelOGS

This program has functions to save the results on files, to read the results in a pandas DataFrame, to compute statistics, and to plot the results. 
"""

#import matplotlib.pyplot as plt
from PySurfaceData.firstModel import *
from scipy.stats import pearsonr
import multiprocessing as mp
import time
import matplotlib.pyplot as plt


print("Number of processors: ", mp.cpu_count())
SEED = 698
torch.manual_seed(SEED)

lambdas = np.array([412.5,442.5,490,510,555]).astype(float)



data = reed_data()
data = data[data['lambda']!=670]
dates = data['date'].drop_duplicates()
#constant = reed_constants('cte_lambda.csv','cst.csv')  ###defined on the model definition

def save_one_result(input_):
    """
    Runs the invertion problem for the bio-optical model, using the loss_function MSELoss(), and the optimizer torch.optim.Adam, 
    then, stores the results in a file in file_+'/'+dates.iloc[date_index].strftime('%Y-%m-%d')+'.csv'. 
    This function is supposed to be used with a DataFrame of data, a DataFrame of dates, and a Dataframe of constants, loaded when importing firstModel.
    Please create the global variables,
    
    >>>data = reed_data()
    >>>data = data[data['lambda']!=670]
    >>>dates = data['date'].drop_duplicates()
    
    data is a panda DataFrame with columns
    date,lambda,RRS,E_dir,E_dif,zenit,PAR.
    dates is a pandas series, with the dates from data on it. 

    input_ is iterable with three elements, the first is the index corresponding to the date in the DataFrame: dates. the second one is the file where to store the data, 
    the third is the number of iterations to be used while training. 

    save_one_result(input_) only stores the result for one date. 
    """
    date_index=input_[0]
    file_=input_[1]
    N=input_[2]
    print(N)
    timeInit=time.time()

    print('starting one process...')
    model = MODEL().to("cpu")
    learning_rate = 1e-3 #this is to use gradient descent. 
    loss_function = nn.MSELoss() #MSE, the same used by Paolo
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #optimizer = torch.optim.LBFGS(model.parameters(),lr=learning_rate) #this requires the closure function, but is much slower. 

    ls_val = []
    ls_count = []
    x = data[data['date'] == dates.iloc[date_index]]
    data_lambdas = x['lambda']
    ls_val,ls_count,pred = train_loop(x,model,loss_function,optimizer,N=N)
    train_values = pd.DataFrame()

    train_values_numpy = pred.detach().numpy()
    
    train_values[str(data_lambdas.iloc[0])] = [train_values_numpy[0]]
    train_values[str(data_lambdas.iloc[1])] = [train_values_numpy[1]]
    train_values[str(data_lambdas.iloc[2])] = [train_values_numpy[2]]
    train_values[str(data_lambdas.iloc[3])] = [train_values_numpy[3]]
    train_values[str(data_lambdas.iloc[4])] = [train_values_numpy[4]]
    train_values['chla'] = [list(model.parameters())[0].detach().numpy()[0][0]]
    train_values['CDOM'] = [list(model.parameters())[1].detach().numpy()[0][0]]
    train_values['NAP'] = [list(model.parameters())[2].detach().numpy()[0][0]]
    train_values['loss'] = [ls_val[-1]]

    train_values.to_csv(file_+'/'+dates.iloc[date_index].strftime('%Y-%m-%d')+'.csv')
    print('time used to save '+ dates.iloc[date_index].strftime('%Y-%m-%d') + ', '+str(time.time() - timeInit)+' seconds')
    return 1

def save_results(init_date_index,end_date_index,cores,file_='results',N=4000):
    
    """
    Runs the invertion problem for the bio-optical model, using the loss_function MSELoss(), and the optimizer torch.optim.Adam, 
    then, stores the results in a file in file_+'/'+dates.iloc[date_index].strftime('%Y-%m-%d')+'.csv'. 
    This function is supposed to be used with a DataFrame of data, a DataFrame of dates, and a Dataframe of constants, loaded when importing firstModel.
    Please create the global variables,
    
    >>>data = reed_data()
    >>>data = data[data['lambda']!=670]
    >>>dates = data['date'].drop_duplicates()
    
    data is a pandas DataFrame with the columns
    date,lambda,RRS,E_dir,E_dif,zenit,PAR.
    dates is a pandas series, with the dates from data on it. 

    save_results uses the function save_one_result to store the data from several dates and parallelize the process in cores number of cores. 
    """
    
    pool = mp.Pool(cores)
    
    pool.map(save_one_result, [(date_index,file_,N) for date_index in range(init_date_index,end_date_index)])
    pool.close()



#print(dates.reset_index()['date'][dates['date']==datetime(year=2006,month=5,day=2)])
#save_results(0,len(dates),40)
save_results(0,1,1,file_='.',N=10)
#plt.plot(np.arange(len(ls_val[0])),ls_val[0])
#plt.show()
#plt.close()

def reed_result(path,data):
    """
    reeds the results stored in path. The files are supposed to be csv files with columns
    ,510.0,412.5,442.5,490.0,555.0,chla,CDOM,NAP,loss

    This function is supposed to be used with a DataFrame of data, a DataFrame of dates, and a Dataframe of constants, loaded when importing firstModel.
    Please create the global variables,
    
    >>>data = reed_data()
    >>>data = data[data['lambda']!=670]
    >>>dates = data['date'].drop_duplicates()
    
    data is a pandas DataFrame with the columns
    date,lambda,RRS,E_dir,E_dif,zenit,PAR.
    dates is a pandas series, with the dates from data on it. 
    
    reed_result reads from all the files on the path, so make sure that the path has no other file than the results. Each result file has the data of one date, and has
    to be stored in a file named %Y-%m-%d.csv. Is meant to be used after storing the data from the save_results function. 

    returns a pandas DataFrame with the columns
    
    """
    results_names = os.listdir(path)
    results = pd.DataFrame(columns=[str(lambdas[0]),str(lambdas[1]),str(lambdas[2]),str(lambdas[3]),str(lambdas[4]),'chla','CDOM','NAP','loss'])
    dates_results = [datetime.strptime(d,'%Y-%m-%d.csv') for d in results_names]
    results = data[data['date'].isin(dates_results)]
    results['RRS_MODEL']=np.empty(len(dates_results)*5)*np.nan
    results['chla']=np.empty(len(dates_results)*5)*np.nan
    results['NAP']=np.empty(len(dates_results)*5)*np.nan
    results['CDOM']=np.empty(len(dates_results)*5)*np.nan
    results['loss']=np.empty(len(dates_results)*5)*np.nan

    
    
    for i in range(len(results_names)):

        d = dates_results[i]
        results_i = pd.read_csv(path+results_names[i],index_col=0)
        results.loc[(results['date']==d) & (results['lambda']==lambdas[0]),'RRS_MODEL'] = float(results_i[str(lambdas[0])])
        results.loc[(results['date']==d) & (results['lambda']==lambdas[1]),'RRS_MODEL'] = float(results_i[str(lambdas[1])])
        results.loc[(results['date']==d) & (results['lambda']==lambdas[2]),'RRS_MODEL'] = float(results_i[str(lambdas[2])])
        results.loc[(results['date']==d) & (results['lambda']==lambdas[3]),'RRS_MODEL'] = float(results_i[str(lambdas[3])])
        results.loc[(results['date']==d) & (results['lambda']==lambdas[4]),'RRS_MODEL'] = float(results_i[str(lambdas[4])])
 

        results.loc[(results['date']==d),'chla'] = float(results_i['chla'])
        results.loc[(results['date']==d),'CDOM'] = float(results_i['CDOM'])
        results.loc[(results['date']==d),'NAP'] = float(results_i['NAP'])
        results.loc[(results['date']==d),'loss'] = float(results_i['loss'])

    return results

        
results = reed_result('results/',data)
print(results)

#plot
def plot_results():
    """
    Plots the RRS from the model and the RRS read from the satellite, both as a function of time. The data is supposed to be stored on a pandas DataFrame with columns
    date,lambda,RRS,E_dir,E_dif,zenit,PAR,chla,NAP,CDOM,loss. The function is made to plot the result in 5 different wavelengths. 
    """
    fig, ax = plt.subplots(2, 3)
    positions = [[0,0],[0,1],[0,2],[1,0],[1,1]]

    for i in range(5):
    
        results_plot=results[results['lambda']==lambdas[i]]
        results_plot = results_plot.sort_values(by='date')

        x_ticks = results_plot['date']
        x_ticks = [(d - x_ticks.iloc[0]).days for d in x_ticks.iloc[:]]

        y_RRS = results_plot['RRS']
        y_RRS_MODEL = results_plot['RRS_MODEL']

        ax[positions[i][0],positions[i][1]].plot(x_ticks,y_RRS,'--',color='black',label='RRS lambda:'+str(lambdas[i]))
        ax[positions[i][0],positions[i][1]].scatter(x_ticks,y_RRS_MODEL,marker='o',c='peachpuff',edgecolors='peru',label='RRS MODEL',alpha=0.5)
        ax[positions[i][0],positions[i][1]].set_xlabel('RRS (sr-1)')
        ax[positions[i][0],positions[i][1]].set_ylabel('Days since 2005-01-02')
        ax[positions[i][0],positions[i][1]].legend()
        
    ax[1,2].axis('off')
    plt.show()
    plt.close()
#plot_results()

#scatter plots
def bias_function(x_data,x_model):
    """
    returns the relative bias between two sets of data.  
    """
    return (np.sum(x_data - x_model)/len(x_data))/np.mean(x_data)

def LSE_function(x_data,x_model):
    """
    returns the relative LSE for two sets of data
    """
    return np.sqrt( np.sum((x_data - x_model)**2)/len(data)  )/np.mean(x_data)

def correlation_function(x_data,x_model):
    """
    returns the Pearson correlation function between two sets of data, using the scipy.stats.pearsonr function. 
    """
    return pearsonr(x_data,x_model)[0]

def scatter_plot():

    """
    Plots the RRS from the model on the x axis and the RRS read from the satellite as the y axis. The data is supposed to be stored on a pandas DataFrame with columns
    date,lambda,RRS,E_dir,E_dif,zenit,PAR,chla,NAP,CDOM,loss. The function is made to plot the result in 5 different wavelengths. 
    """
    fig, ax = plt.subplots(2, 3)
    positions = [[0,0],[0,1],[0,2],[1,0],[1,1]]
    colors = ['deepskyblue','lightcoral','limegreen','silver','pink']
    for i in range(5):
        results_plot=results[results['lambda']==lambdas[i]]
        results_plot = results_plot.sort_values(by='date')

        #x_ticks = results_plot['date']
        #x_ticks = [(d - x_ticks.iloc[0]).days for d in x_ticks.iloc[:]]

        y_RRS = results_plot['RRS']
        y_RRS_MODEL = results_plot['RRS_MODEL']

        bias = bias_function(y_RRS,y_RRS_MODEL)
        LSE = LSE_function(y_RRS,y_RRS_MODEL)
        correlation = correlation_function(y_RRS,y_RRS_MODEL)
        label = 'lambda: {:.4f}\nRelative bias: {:.4f}\nRelative Standard Deviation: {:.4f}\nCorrelation: {:.4f}'.format(lambdas[i],bias,LSE,correlation)
        ax[positions[i][0],positions[i][1]].scatter(y_RRS,y_RRS_MODEL,marker='o',c=colors[i],label=label,alpha=0.6,edgecolors='black')
        ax[positions[i][0],positions[i][1]].plot(y_RRS_MODEL,y_RRS_MODEL,'--',color='black',label='Perfect correlation line.')
        ax[positions[i][0],positions[i][1]].legend()
        ax[positions[i][0],positions[i][1]].set_xlabel('RRS_DATA (sr-1)')
        ax[positions[i][0],positions[i][1]].set_ylabel('RRS_MODEL (sr-1)')
    ax[1,2].axis('off')
    plt.show()
    plt.close()
#scatter_plot()

def self_correlation(dat,column):
    """
    computes the time self correlation. dat is supposed to be a pandas DataFrame, so the column is the name of the column with the data, which self-correlation is intended to be computed.
    """
    work_data = np.empty((5,int(len(dat)/5-1)))  #I want a matrix win 5 rows and as many columns as days,  minus one, because I will compare it with the shifted data, so I will no use the first data. 
    shift_data = np.empty((5,int(len(dat)/5-1)))
    for i in range(5):
        work_data[i] = dat[column][dat['lambda'] == lambdas[i]].iloc[1:].to_numpy()
        shift_data[i] = dat[column][dat['lambda'] == lambdas[i]].shift(1).iloc[1:].to_numpy()
    X = work_data - np.nanmean(work_data,axis=0)
    Y = shift_data - np.nanmean(shift_data,axis=0)
    cov = np.nanmean(X*Y,axis=0)
    sigma_x = np.nanstd(X,axis=0)
    sigma_y = np.nanstd(Y,axis=0)
    corr = cov/(sigma_x*sigma_y)
    return np.nanmean(corr)
    
        
        
#print(self_correlation(data,'RRS'))
#print(self_correlation(results,'RRS_MODEL'))

#plt.plot(np.arange(len(results[results['lambda']==lambdas[0]])),results[results['lambda']==lambdas[0]]['CDOM'],'o')
#plt.show()
print(results['zenit'][results['zenit']!=0])

buoy_data=pd.read_csv('/g100_work/OGS21_PRACE_P/plazzari/Forward_Adjoint/BOUSSOLE_DATA/orig/buoy.DPFF.2003-09-06_2012-12-31_999.dat',sep='\t')


buoy_results = np.empty((len(results)))
for i in range(len(results)):
        date = results['date'].iloc[i]
        buoy_results[i] = buoy_data[ (buoy_data['YEAR'] == date.year) & (buoy_data['MONTH'] == date.month) & (buoy_data['DAY'] == date.day) ]['chl'].mean()
results['buoy_chla'] = buoy_results

#print(self_correlation(data,'RRS'))
#print(self_correlation(results,'RRS_MODEL'))

#plt.plot(np.arange(len(results[results['lambda']==lambdas[0]])),results[results['lambda']==lambdas[0]]['CDOM'],'o')
#plt.show()
#print(results['zenit'][results['zenit']!=0])


def plot_chl():
    fig, ax = plt.subplots(1,1)


    for i in range(1):

        results_plot=results[results['lambda']==lambdas[i]]
        results_plot = results_plot.sort_values(by='date')

        x_ticks = results_plot['date']
        x_ticks = [(d - x_ticks.iloc[0]).days for d in x_ticks.iloc[:]]
        x_ticks_lab = [datetime.strftime(dat,'%Y-%m-%d') for dat in results_plot['date'].iloc[:] ]
        ax.set_xticks(x_ticks[::200],x_ticks_lab[::200])

        y_chl = results_plot['buoy_chla']
        y_chl_MODEL = results_plot['chla']
        print(y_chl)
        print(y_chl_MODEL)
        ax.plot(x_ticks,y_chl,'--',color='black',label='chl data')
        ax.scatter(x_ticks,y_chl_MODEL,marker='o',c='yellowgreen',edgecolors='darkolivegreen',label='chla MODEL',alpha=0.5)
        ax.set_ylabel('chl (mg/m^3)')
        ax.set_xlabel('Date')
        ax.set_yscale('log')
        ax.legend()

    plt.show()
    plt.close()
#plot_chl()
