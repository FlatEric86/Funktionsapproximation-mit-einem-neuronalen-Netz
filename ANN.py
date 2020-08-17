
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F

import random as rand
import os
import time as t

import matplotlib.pyplot as plt


plt.style.use('ggplot')
    
   




   
df      = pd.read_csv('./t_data.csv')
df_orig = pd.read_csv('./ideal_data.csv')
df      = df.sample(frac=1).reset_index(drop=True)




X = torch.tensor([[x] for x in np.asarray(df['X'])]).float()
Y = torch.tensor([[0.1*y] for y in np.asarray(df['Y'])]).float()

# check if CUDA is possible and store value
use_cuda = torch.cuda.is_available()

use_cuda = False

# define device in dependency if CUDA is available or not. 
device   = torch.device('cuda:0' if use_cuda else 'cpu')



# define number of CPU cores if device gots defined as CPU
if use_cuda == False:
    torch.set_num_threads(14)   # let one core there for os
    
    
    
# splitte die Trainingsdaten zu je (ca.) 50 % in tatsächlichen Trainingsdaten und Validierungsdaten
X_training = X[:len(X)//2]
X_training = X_training.to(device) 

X_valid    = X[len(X)//2:]
X_valid    = X_valid.to(device) 

Y_training = Y[:len(Y)//2]
Y_training = Y_training.to(device) 

Y_valid    = Y[len(Y)//2:]
Y_valid    =  Y_valid.to(device) 





class model(nn.Module):

    def __init__(self):
        super(model, self).__init__()
        
        n = 100
        
        self.lin1 = nn.Linear(1, n)
        self.lin2 = nn.Linear(n, n)
        #self.lin3 = nn.Linear(n, n)
        #self.lin4 = nn.Linear(n, n)
        #self.lin5 = nn.Linear(n, n)
        #self.lin6 = nn.Linear(n, n)
        #self.lin7 = nn.Linear(n, n)
        self.lin8 = nn.Linear(n, 1)

        # self.lin3 = nn.Linear(len(X[0]), 1)        
        
        
        
    def forward(self, x):
    
        x = self.lin1(x)
        x = torch.nn.functional.tanh(self.lin2(x))
        #x = torch.nn.functional.tanh(self.lin3(x))
        #x = torch.nn.functional.sigmoid(self.lin4(x))
        #x = torch.nn.functional.sigmoid(self.lin5(x))
        #x = torch.nn.functional.sigmoid(self.lin6(x))
        #x = torch.nn.functional.sigmoid(self.lin7(x))
        x = torch.nn.functional.tanh(self.lin8(x))
        
        return x
        
      
      
N_epoch = 15000
lr      = 0.0005  
      

criterion = nn.MSELoss()      
netz = model().to(device)
    
    
    
if os.path.isfile('./weights.pt'):
    netz.load_state_dict(torch.load('./weights.pt'))      
    
    
#optimizer = torch.optim.SGD(netz.parameters(), lr=lr)
optimizer = torch.optim.Adam(netz.parameters(), lr=lr)


loss_function = nn.MSELoss()   
        
        
      
EPOCH   = []
LOSS    = []        
VALID   = []  
        
for i in range(N_epoch):



    ### set gradient as 0 
    netz.zero_grad()
    
    
    outputs = netz(X_training).to(device)
    
    loss  = loss_function(outputs, Y_training)
    
    valid_model = netz(X_valid).to(device)
    
    valid = loss_function(valid_model, Y_valid)
    print(80*'~')
    print('EPOCH:',i)
    print(float(loss))
    

    
    loss.backward()
    optimizer.step()

    
    EPOCH.append(i)
    LOSS.append(float(loss))
    VALID.append(float(valid))


    


    ## do optimize the model
    optimizer = torch.optim.SGD(netz.parameters(), lr=lr)
    optimizer.step()


torch.save(netz.state_dict(), './weights.pt')


plt.plot(EPOCH, LOSS, color='green', label='model')
plt.plot(EPOCH, VALID, color='blue', label='test')
plt.legend()
plt.title('Evolution der Modelloptimierung')
plt.xlabel('Anzahl der Lernepochen')
plt.ylabel('Mittlerer quadratischer Fehler')
plt.show()

plt.close()


Y_model = netz(torch.tensor([[x] for x in df_orig['X'].tolist()])).tolist()

plt.plot(df_orig['X'], 0.1*df_orig['Y'], label='origin function', color='k')
plt.scatter(X_training, Y_training, label='origin fun + noise\n(training)', color='green', marker='.')
plt.scatter(X_valid, Y_valid, label='origin fun + noise\n(test)', color='orange', marker='.')
plt.plot(df_orig['X'], Y_model, label='model', color='blue')

plt.title('Modell nach ' 
    + str(N_epoch) 
    + ' Lernepochen'
)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()






