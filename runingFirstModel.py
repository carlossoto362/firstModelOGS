
import matplotlib.pyplot as plt
from firstModel import *


SEED = 698
torch.manual_seed(SEED)

lambdas = np.array([412.5,442.5,490,510,555]).astype(float)



data = reed_data()
data = data[data['lambda']!=670]
dates = data['date'].drop_duplicates()
#constant = reed_constants('cte_lambda.csv','cst.csv')  ###defined on the model definition

def save_results(dates,data):
    ls_val_all = []
    for i in range(len(dates)):

        print(dates.iloc[i].strftime('%Y-%m-%d'))

        model = MODEL().to("cpu")
        learning_rate = 1e-2 #this is to use gradient descent. 
        loss_function = nn.MSELoss() #MSE, the same used by Paolo
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        #optimizer = torch.optim.LBFGS(model.parameters(),lr=learning_rate) #this requary the closure function, but is much slower. 

        N=1000#number of iterations
        ls_val = []
        ls_count = []

        ls_val,ls_count,pred = train_loop(data[data['date'] == dates.iloc[i]],model,loss_function,optimizer,N=5001)
        train_values = pd.DataFrame()

        train_values['RRS_1'] = [pred.detach().numpy()[0]]
        train_values['RRS_2'] = [pred.detach().numpy()[1]]
        train_values['RRS_3'] = [pred.detach().numpy()[2]]
        train_values['RRS_4'] = [pred.detach().numpy()[3]]
        train_values['RRS_5'] = [pred.detach().numpy()[4]]
        train_values['chla'] = [list(model.parameters())[0].detach().numpy()[0][0]]
        train_values['CDOM'] = [list(model.parameters())[1].detach().numpy()[0][0]]
        train_values['NAP'] = [list(model.parameters())[2].detach().numpy()[0][0]]
        train_values['loss'] = [ls_val[-1]]

        train_values.to_csv('results/'+dates.iloc[i].strftime('%Y-%m-%d')+'.csv')
        ls_val_all.append(ls_val)
    return ls_val_all
        
ls_val = save_results(dates.iloc[:1],data)
plt.plot(np.arange(len(ls_val[0])),ls_val[0])
plt.show()
plt.close()

def reed_result(path,data):
    results_names = os.listdir(path)
    results = pd.DataFrame(columns=['RRS_1','RRS_2','RRS_3','RRS_4','RRS_5','chla','CDOM','NAP','loss'])
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
        results.loc[(results['date']==d) & (results['lambda']==lambdas[0]),'RRS_MODEL'] = float(results_i['RRS_1'])
        results.loc[(results['date']==d) & (results['lambda']==lambdas[1]),'RRS_MODEL'] = float(results_i['RRS_2'])
        results.loc[(results['date']==d) & (results['lambda']==lambdas[2]),'RRS_MODEL'] = float(results_i['RRS_3'])
        results.loc[(results['date']==d) & (results['lambda']==lambdas[3]),'RRS_MODEL'] = float(results_i['RRS_4'])
        results.loc[(results['date']==d) & (results['lambda']==lambdas[4]),'RRS_MODEL'] = float(results_i['RRS_5'])
 

        results.loc[(results['date']==d),'chla'] = float(results_i['chla'])
        results.loc[(results['date']==d),'CDOM'] = float(results_i['CDOM'])
        results.loc[(results['date']==d),'NAP'] = float(results_i['NAP'])
        results.loc[(results['date']==d),'loss'] = float(results_i['loss'])

    return results

        
results = reed_result('results/',data)
print(results)

#plot

results_plot=results[results['lambda']==lambdas[0]]
results_plot = results_plot.sort_values(by='date')

x_ticks = results_plot['date']
x_ticks = [(d - x_ticks.iloc[0]).days for d in x_ticks.iloc[:]]

y_RRS = results_plot['RRS']
y_RRS_MODEL = results_plot['RRS_MODEL']

plt.plot(x_ticks,y_RRS,'--',color='blue',label='RRS')
plt.plot(x_ticks,y_RRS_MODEL,'o',color='black',label='RRS MODEL')
plt.legend()
plt.show()
