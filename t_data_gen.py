import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand


def fun(x, a, b, c, noise=None):

    if noise == True:
        y   = a*x**2 + b*x + c

        return y + rand.choice([-1, 1])*rand.gauss(0, 0.5) 
    else:
        return a*x**2 + b*x + c 
   

   
def fun_sin(x, noise=None):
    
    if noise == True:
        y   = np.exp(-0.08*x)*np.sin(x) 
        
        
        return y + rand.choice([-1, 1])*rand.gauss(0, 0.2*np.exp(-0.08*x))
    
    elif noise==False:
        return np.exp(-0.08*x)*np.sin(x)
        
    
    
def stair_fun(X, noise=None):
    Y = []
    
    i = 0
    for x in X:
        if x % 10 == 0:
            i += 0.1
            print(i)
        
        if noise == True:
            Y.append(i + rand.choice([-1, 1])*rand.gauss(0, 0.01))
        else:
            Y.append(i)
        
    return Y
    
    
    
a = -.01
b = -.09
c = -.03   
d = 0    
    
X = np.arange(-5, 5, 0.07)

#Y       = [fun(x, a, b, c, noise=True) for x in X]
#Y_ideal = [fun(x, a, b, c, noise=False) for x in X]

Y       = [fun_sin(x, noise=True) for x in X]
Y_ideal = [fun_sin(x, noise=False) for x in X]

#Y        = stair_fun(X, noise=False)
#Y_ideal  = stair_fun(X, noise=True)


#print(Y_ideal)

df       = pd.DataFrame(columns=['X', 'Y'])
df_ideal = pd.DataFrame(columns=['X', 'Y'])

df['X']       = X
df['Y']       = Y
df_ideal['X'] = X
df_ideal['Y'] = Y_ideal

df.to_csv('./t_data.csv', index=True)
df_ideal.to_csv('./ideal_data.csv', index=True)



plt.plot(X, Y, color='green')
plt.plot(X, Y_ideal, color='blue')
plt.show()

    
    
    