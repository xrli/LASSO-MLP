#!/usr/bin/env python
# coding: utf-8

# In[28]:


from astropy.io import fits
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
from scipy import sparse
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from sys import argv
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import os, gc
import math
from tqdm import tqdm
# from scikits.sparse.cholmod import cholesky
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


data = pd.read_csv("../DR820_30_唯一交叉匹配.csv")


# In[35]:


data


# In[40]:


data['combined_obsid'][1604]


# In[44]:


data['FeH'][1603]


# In[3]:


wavelength_new = np.load('../拟合方法_多项式/wavelength_new_un.npy')


# In[ ]:


-0.052000000000000005


# In[4]:


flux_list1 = []
wavelengths_list1 = []
flux_list = []
wavelengths_list = []
min_max = []
flux_interp1d_list = []

valid_index0 = np.where(np.nan_to_num(data['combined_teff']) != -9999.99)[0]
param1 = data.iloc[valid_index0]

valid_index0 = np.where(np.nan_to_num(data['combined_logg']) != -9999.99)[0]
param1 = data.iloc[valid_index0]


valid_index = np.where(param1['combined_z'] != -9999)[0] 
valid_param = param1.iloc[valid_index]


# In[5]:


c2 = np.load('../拟合方法_多项式/fit_多项式_low_new.npy')
train_labe1 = valid_param['Teff[K]']
train_labe2 = valid_param['Logg']
train_labe3 = valid_param['FeH']
train_labe1 = np.log10(train_labe1)


# In[6]:


c2 = np.load('../拟合方法_多项式/fit_多项式_low_new.npy')


# In[7]:


np.shape(c2)


# In[19]:


#X_train0, X_test0, y_train0, y_test0 = train_test_split(c0, label, test_size=0.2, random_state=42)|
X_train, X_val, y_train, y_val = train_test_split(c2,train_labe1, test_size=0.2, random_state=42)


# In[20]:


np.shape(X_train)


# In[21]:


np.shape(X_val)


# In[22]:


from sklearn import svm
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

sc=StandardScaler()
target_sc = sc.fit(X_train)#计算样本的均值和标准差
X_train_std=target_sc.transform(X_train)
X_val_std=target_sc.transform(X_val)


from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV 
clf = Lasso(alpha = 0.0003,max_iter=10000,tol=0.00001)
#0.0003

clf.fit(X_train_std,y_train)
mask = clf.coef_ != 0
X_1 = X_train_std[:,mask]

X_2 = X_val_std[:,mask]
print(X_2.shape)


# In[ ]:





# In[23]:


clf.coef_


# In[24]:


clf.coef_[clf.coef_ != 0] = 1
clf.coef_


# In[25]:


# clf2 = MLPRegressor(hidden_layer_sizes=(100,60,60,20),activation='tanh',solver='lbfgs',  alpha=0.05, batch_size=80,
#                          learning_rate='adaptive', learning_rate_init=0.001, power_t=0.001
#         , max_iter=10000, shuffle=True,
#                          random_state=42, tol=0.00001, verbose=True, warm_start=True, nesterovs_momentum=True,
#                          early_stopping=True,beta_1=0.999, beta_2=0.99, epsilon=1e-7)
clf2 = MLPRegressor(hidden_layer_sizes=(80,80,40),activation='tanh',solver='lbfgs',  alpha=0.2, batch_size=60,
                         learning_rate='adaptive', learning_rate_init=0.001, power_t=0.001
        , max_iter=10000, shuffle=True,
                         random_state=42, tol=0.000001, verbose=True, warm_start=True, nesterovs_momentum=True,
                         early_stopping=True,beta_1=0.99, beta_2=0.99, epsilon=1e-7)
clf2.fit(X_1, y_train),

y_pred2 = clf2.predict(X_2)

#print(y_pred)


# In[26]:


print(mean_absolute_error (10 ** y_val, 10 ** y_pred2))
x1 = 10 ** y_pred2 - 10 ** y_val
mu =np.mean(x1) #计算均值 
sigma =np.std(x1) 
print(mu)
print(sigma)


# In[27]:


teff_error = 10 ** y_pred2 - 10 ** y_val

np.save('./teff_error.npy',teff_error)
np.save('./teff_val.npy',10 ** y_val)
np.save('./teff_pred.npy',10 ** y_pred2)


# In[ ]:





# In[ ]:





# In[28]:


# clf2 = MLPRegressor(hidden_layer_sizes=(80,80,40),activation='tanh',solver='lbfgs',  alpha=0.2, batch_size=60,
#                          learning_rate='adaptive', learning_rate_init=0.001, power_t=0.001
#         , max_iter=10000, shuffle=True,
#                          random_state=42, tol=0.000001, verbose=True, warm_start=True, nesterovs_momentum=True,
#                          early_stopping=True,beta_1=0.99, beta_2=0.99, epsilon=1e-7)
# 84.32583367100601
# 0.20584463408454118
# 164.89734805273315


# In[29]:


# clf2 = MLPRegressor(hidden_layer_sizes=(60,60,40),activation='tanh',solver='lbfgs',  alpha=0.2, batch_size=80,
#                          learning_rate='adaptive', learning_rate_init=0.001, power_t=0.001
#         , max_iter=10000, shuffle=True,
#                          random_state=42, tol=0.000001, verbose=True, warm_start=True, nesterovs_momentum=True,
#                          early_stopping=True,beta_1=0.99, beta_2=0.99, epsilon=1e-9)


# 83.74182541931755
# -1.327234741446021
# 148.75836110853237 


# In[30]:


import seaborn as sns
plt.figure(figsize=(5, 5))

h = sns.jointplot(10 ** y_val,10 ** y_pred2,kind = 'reg')
#'scatter', 'reg', 'resid', 'kde', or 'hex'
#h.plot_joint(sns.kdeplot, zorder=0, n_levels=6,color='red')
# JointGrid has a convenience function
h.set_axis_labels('x', 'y', fontsize=15)

plt.grid(True)
 
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)   
    
    
    

# or set labels via the axes objects
h.ax_joint.set_xlabel('$\mathrm{T}_{\mathrm{eff}}^{\mathrm{APOGEE}}$ (K)')
h.ax_joint.set_ylabel('$\mathrm{T}_{\mathrm{eff}}^{\mathrm{LASSO-MLP}}$ (K)')
plt.tight_layout()


#plt.tick_params(top=True,bottom=False,left=False,right=False)
#plt.tick_params(labeltop=True,labelleft=True,labelright=True,labelbottom=True)

plt.savefig('2_6_teff_线性_apogee.pdf')


# In[31]:


x = 10 ** y_pred2 - 10 ** y_val


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import matplotlib.mlab as mlab
from scipy.stats import norm
 
#n, bins, patches = plt.hist(x, 20, density=1, facecolor='blue', alpha=0.75)  #第二个参数是直方图柱子的数量
mu =np.mean(x) #计算均值 
sigma =np.std(x) 
num_bins = 300 #直方图柱子的数量 
n, bins, patches = plt.hist(x, num_bins,density=1) 

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

#直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象 
y = norm.pdf(bins, mu, sigma)#拟合一条最佳正态分布曲线y 
#plt.grid(True)
plt.plot(bins, y, 'r--',label = '$\mu = $'+ str(round(mu,4)) + "\n"+ '$\sigma= $'+str(round(sigma,3)))
plt.xlabel('Delta $\mathrm{T}_{\mathrm{eff}}$ (K)',fontsize=15) 
plt.ylabel('Frequency',fontsize=15) #绘制y轴 
#plt.title('Histogram : $\mu$=' + str(round(mu,4)) + ' $\sigma=$'+str(round(sigma,3)))  #中文标题 u'xxx' 
#plt.subplots_adjust(left=0.15)#左边距 
#plt.savefig('20-30TeffA-P高斯.pdf')
plt.legend(loc='upper left',frameon=False,fontsize = 15)
plt.savefig('2_6_teff_高斯_apogee.eps')


# In[33]:


import seaborn as sns
teff_error = 10 ** y_pred2 - 10 ** y_val
plt.figure(figsize=(10, 5))
plt.scatter(10 ** y_val,teff_error)
plt.xlabel('$\mathrm{\mathrm{T}_{\mathrm{eff}}}^{\mathrm{APOGEE}}$ (K)',fontsize=15)
plt.ylabel('Delta $\mathrm{T}_{\mathrm{eff}}$ (K)',fontsize=15)

plt.axhline(y=0, color='r', linestyle='-')
#plt.tick_params(top=True,bottom=False,left=False,right=False)
#plt.tick_params(labeltop=True,labelleft=True,labelright=True,labelbottom=True)

plt.savefig('2_6_teff_apogee_error.eps')


# In[ ]:





# In[ ]:





# In[34]:


#X_train0, X_test0, y_train0, y_test0 = train_test_split(c0, label, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(c2,train_labe2, test_size=0.2, random_state=42)


# In[35]:


from sklearn import svm
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

sc=StandardScaler()
sc.fit(X_train)#计算样本的均值和标准差
X_train_std=sc.transform(X_train)
X_val_std=sc.transform(X_val)

from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV 
clf = Lasso(alpha=0.005,max_iter=10000)
# clf = Lasso(alpha = 0.003,fit_intercept=True, normalize=False, precompute=True, 
#             copy_X=True, max_iter=10000, tol=1e-4, warm_start=False, 
#             positive=False, random_state=None)
#0.01
clf.fit(X_train_std,y_train)
mask = clf.coef_ != 0
X_1 = X_train_std[:,mask]
print(X_1.shape)
X_2 = X_val_std[:,mask]


# In[36]:


clf.coef_[clf.coef_ != 0] = 1
clf.coef_


# In[37]:


model_mlp = MLPRegressor(hidden_layer_sizes=(180,150,120,80,80,60,60,40,20),activation='relu',solver='adam', alpha=0.1, batch_size=100,
                         learning_rate='adaptive', power_t=0.01, max_iter=10000, shuffle=True, learning_rate_init=0.004,
                         random_state=42, tol=0.000001, verbose=False, warm_start=False, nesterovs_momentum=True,
                         early_stopping=True,beta_1=0.99, beta_2=0.999, epsilon=1e-7)
# model_mlp = MLPRegressor(hidden_layer_sizes=(40,10,5),activation='relu',solver='lbfgs', alpha=0.001, batch_size=200,
#                          learning_rate='adaptive', power_t=0.01, max_iter=1000, shuffle=True,
#                          random_state=1, tol=0.001, verbose=False, warm_start=False, nesterovs_momentum=True,
#                          early_stopping=True,beta_1=0.999, beta_2=0.999, epsilon=1e-7)
model_mlp.fit(X_1,y_train)
y_pred=model_mlp.predict(X_2)
#print(clf.alpha_)


# In[38]:


print('MAE log g:', mean_absolute_error(y_pred,y_val))
x1 = y_pred - y_val
mu =np.mean(x1) #计算均值 
sigma =np.std(x1) 
print(mu)
print(sigma)


# In[39]:


logg_error = x = y_pred - y_val

np.save('./logg_error.npy',logg_error)
np.save('./logg_val.npy',y_val)
np.save('./logg_pred.npy',y_pred)


# In[40]:


model_mlp = MLPRegressor(hidden_layer_sizes=(180,150,120,80,80,60,60,40,20),activation='relu',solver='adam', alpha=0.1, batch_size=100,
                         learning_rate='adaptive', power_t=0.01, max_iter=10000, shuffle=True, learning_rate_init=0.004,
                         random_state=42, tol=0.000001, verbose=False, warm_start=False, nesterovs_momentum=True,
                         early_stopping=True,beta_1=0.99, beta_2=0.99, epsilon=1e-7)

0.13990988553552228
-0.00017533431645589266
0.23499572564888022


# In[ ]:





# In[ ]:





# In[41]:


import seaborn as sns
plt.figure(figsize=(5, 5))

h = sns.jointplot(y_val,y_pred,kind = 'reg')
#'scatter', 'reg', 'resid', 'kde', or 'hex'
#h.plot_joint(sns.kdeplot, zorder=0, n_levels=6,color='red')
# JointGrid has a convenience function
h.set_axis_labels('x', 'y', fontsize=15)

plt.grid(True)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# or set labels via the axes objects
h.ax_joint.set_xlabel('log $\mathrm{g}_{\mathrm{APOGEE}}$ (dex)',fontsize = 15)
h.ax_joint.set_ylabel('log $\mathrm{g}_{\mathrm{LASSO-MLP}}$ (dex)',fontsize = 15)
plt.tight_layout()

#plt.tick_params(top=True,bottom=False,left=False,right=False)
#plt.tick_params(labeltop=True,labelleft=True,labelright=True,labelbottom=True)

plt.savefig('2_6_logg_线性_apogee.pdf')


# In[42]:


x = y_pred - y_val
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import matplotlib.mlab as mlab
from scipy.stats import norm
 
 

#n, bins, patches = plt.hist(x, 20, density=1, facecolor='blue', alpha=0.75)  #第二个参数是直方图柱子的数量
mu =np.mean(x) #计算均值 
sigma =np.std(x) 
num_bins = 300 #直方图柱子的数量 
n, bins, patches = plt.hist(x, num_bins,density=1) 
#直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象 
y = norm.pdf(bins, mu, sigma)#拟合一条最佳正态分布曲线y 
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.grid(True)
plt.plot(bins, y, 'r--',label = '$\mu = $'+ str(round(mu,4)) + "\n"+ '$\sigma= $'+str(round(sigma,3)))
plt.xlabel('Delta log $\mathrm{g}$ (dex)',fontsize=15) 
plt.ylabel('Frequency',fontsize=15) #绘制y轴 
#plt.title('Histogram : $\mu$=' + str(round(mu,4)) + ' $\sigma=$'+str(round(sigma,3)))  #中文标题 u'xxx' 
#plt.subplots_adjust(left=0.15)#左边距 
#plt.savefig('20-30TeffA-P高斯.pdf')
plt.legend(loc='upper left',frameon=False,fontsize = 15)
plt.savefig('2_6_logg_高斯_apogee.eps')


# In[43]:


import seaborn as sns
logg_error =  y_pred - y_val
plt.figure(figsize=(10, 5))
plt.scatter(y_val,logg_error)
plt.xlabel('log $\mathrm{g}^{\mathrm{APOGEE}}$ (dex)',fontsize=15)
plt.ylabel('Delta log $\mathrm{g}$ (dex)',fontsize=15)

plt.axhline(y=0, color='r', linestyle='-')
#plt.tick_params(top=True,bottom=False,left=False,right=False)
#plt.tick_params(labeltop=True,labelleft=True,labelright=True,labelbottom=True)

plt.savefig('2_6_logg_apogee_error.eps')


# In[ ]:





# In[8]:


#X_train0, X_test0, y_train0, y_test0 = train_test_split(c0, label, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(c2,train_labe3, test_size=0.2, random_state=42)


# In[9]:


from sklearn import svm
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

sc=StandardScaler()
sc.fit(X_train)#计算样本的均值和标准差
X_train_std=sc.transform(X_train)
X_val_std=sc.transform(X_val)

from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV 
clf = Lasso(alpha=0.001,max_iter=10000)
# clf = Lasso(alpha = 0.001, fit_intercept=True, normalize=False, precompute=True, 
#             copy_X=True, max_iter=10000, tol=1e-4, warm_start=False, 
#             positive=False, random_state=None
#            )
#alpha = 0.01

clf.fit(X_train_std,y_train)
mask = clf.coef_ != 0
X_1 = X_train_std[:,mask]
#X_1 = X_train_std
print(X_1.shape)
X_2 = X_val_std[:,mask]
#X_2 = X_val_std


# In[10]:


model_mlp= MLPRegressor(hidden_layer_sizes=(300,200,100,80,80,40,40,20),activation='tanh',solver='lbfgs', alpha=0.1, batch_size=80,
                         learning_rate='constant', power_t=0.001, max_iter=10000, shuffle=True,
                         random_state=42, tol=0.0001, verbose=False, warm_start=False, nesterovs_momentum=True,
                         early_stopping=True,beta_1=0.999, beta_2=0.999, epsilon=1e-6)
model_mlp.fit(X_1,y_train)
y_pred=model_mlp.predict(X_2)
#print(clf.alpha_)
print('MAE feh:', mean_absolute_error(y_pred,y_val))

x1 = y_pred - y_val
mu =np.mean(x1) #计算均值 
sigma =np.std(x1) 
print(mu)
print(sigma)


# In[47]:


feh_error = y_pred - y_val
np.save('./feh_error.npy',feh_error)
np.save('./feh_val.npy',y_val)
np.save('./feh_pred.npy',y_pred)


# In[3]:


teff_val = np.load('teff_val.npy')
teff_pred = np.load('teff_pred.npy')
teff_error = np.load('teff_error.npy')

logg_val = np.load('logg_val.npy')
logg_pred = np.load('logg_pred.npy')
logg_error = np.load('logg_error.npy')

feh_val = np.load('feh_val.npy')
feh_pred = np.load('feh_pred.npy')
feh_error = np.load('feh_error.npy')


# In[8]:


np.min(feh_error)


# In[19]:


a=np.where(teff_error==np.max(teff_error))
b=np.where(teff_error==np.min(teff_error))
c=np.where(logg_error==np.max(logg_error))
d=np.where(logg_error==np.min(logg_error))
e=np.where(feh_error==np.max(feh_error))
f=np.where(feh_error==np.min(feh_error))
g=np.where(feh_error<=-0.6)


# In[47]:


g


# In[22]:


feh_pred[1604]-feh_val[1604]


# In[48]:


print(feh_val[1604])


# In[49]:


h = np.where(data['FeH']==-0.052000000000000005)[0]


# In[50]:


h


# In[56]:


print(data['combined_obsid'][830])
print(data['combined_obsid'][1350])
print(data['combined_obsid'][1713])
print(data['combined_obsid'][2537])
print(data['combined_obsid'][5539])
print(data['combined_obsid'][6529])
print(data['combined_obsid'][7016])
print(data['combined_obsid'][7164])
print(data['combined_obsid'][8955])
print(data['combined_obsid'][8991])
print(data['combined_obsid'][9716])
print(data['combined_obsid'][9733])
print(data['combined_obsid'][10066])


# In[57]:


d8 = pd.read_csv("../LASSO-MLP与SMLP模型估计结果的融合.csv")


# In[62]:


h = np.where(d8['combined_obsid']==52905206)[]


# In[64]:


h[82835]


# In[ ]:





# In[51]:


print(teff_pred[964]-teff_val[964])
print(teff_pred[1471]-teff_val[1471])
print(logg_pred[1109]-logg_val[1109])
print(logg_pred[1446]-logg_val[1446])
print(feh_pred[671]-feh_val[671])
print(feh_pred[1471]-feh_val[1471])


# In[52]:


print(teff_val[964])
print(teff_val[1471])
print(logg_val[1109])
print(logg_val[1446])
print(feh_val[671])
print(feh_val[1471])


# In[53]:


print(y_val.iloc[964]['combined_file'])
print(y_val.iloc[1471]['combined_file'])
print(y_val.iloc[1109]['combined_file'])
print(y_val.iloc[1446]['combined_file'])
print(y_val.iloc[671]['combined_file'])
print(y_val.iloc[1471]['combined_file'])


# In[ ]:


print(y_val.iloc[964]['combined_obsid'])
print(y_val.iloc[1471]['combined_obsid'])
print(y_val.iloc[1109]['combined_obsid'])
print(y_val.iloc[1446]['combined_obsid'])
print(y_val.iloc[671]['combined_obsid'])
print(y_val.iloc[1471]['combined_obsid'])


# In[ ]:


print(np.where(data['combined_obsid'] == 127712034)[0])
print(np.where(data['combined_obsid'] == 403132)[0])
print(np.where(data['combined_obsid'] == 146904177)[0])
print(np.where(data['combined_obsid'] == 507107103)[0])
print(np.where(data['combined_obsid'] == 602714024)[0])
print(np.where(data['combined_obsid'] == 403132)[0])


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(c2,data, test_size=0.2, random_state=42)


# In[ ]:





# In[24]:





# In[25]:





# In[ ]:


data


# In[19]:





# In[20]:


a


# In[15]:


b = np.where(data['Teff[K]']==7581.994999999999)[0]


# In[ ]:


b


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


model_mlp= MLPRegressor(hidden_layer_sizes=(80,80,60,20),activation='tanh',solver='lbfgs', alpha=0.1, batch_size=80,
                         learning_rate='constant', power_t=0.0001, max_iter=10000, shuffle=True,
                         random_state=42, tol=0.0001, verbose=False, warm_start=False, nesterovs_momentum=True,
                         early_stopping=True,beta_1=0.99, beta_2=0.99, epsilon=1e-5)

0.06533966151754969
-0.00034407331103956837
0.09792790694514131

 ## model_mlp= MLPRegressor(hidden_layer_sizes=(300,200,100,80,80,40,40,20),activation='tanh',solver='lbfgs', alpha=0.1, batch_size=80,
                         learn2ing_rate='constant', power_t=0.001, max_iter=10000, shuffle=True,
                         random_state=42, tol=0.0001, verbose=False, warm_start=False, nesterovs_momentum=True,
                         early_stopping=True,beta_1=0.999, beta_2=0.999, epsilon=1e-6)
0.06398025886725621
-0.0003524226257866421
0.0958718917214544



model_mlp= MLPRegressor(hidden_layer_sizes=(300,200,100,80,80,40,40,20),activation='tanh',solver='lbfgs', alpha=0.2, batch_size=80,
                         learning_rate='constant', power_t=0.001, max_iter=10000, shuffle=True,
                         random_state=42, tol=0.0001, verbose=False, warm_start=False, nesterovs_momentum=True,
                         early_stopping=True,beta_1=0.999, beta_2=0.999, epsilon=1e-6)
0.062308009350597895
-0.00098022309382782
0.0956554521644515


model_mlp= MLPRegressor(hidden_layer_sizes=(300,160,120,80,40,20),activation='relu',solver='lbfgs', alpha=0.01, batch_size=80,
                         learning_rate='constant', power_t=0.01, max_iter=10000, shuffle=True,
                         random_state=42, tol=0.0001, verbose=False, warm_start=False, nesterovs_momentum=True,
                         early_stopping=True,beta_1=0.99, beta_2=0.99, epsilon=1e-5)
 0.06341136136509004
0.0017006269755990933
0.09991230409954881


model_mlp= MLPRegressor(hidden_layer_sizes=(300,160,120,80,40,20),activation='relu',solver='lbfgs', alpha=0.001, batch_size=80,
                         learning_rate='constant', power_t=0.01, max_iter=10000, shuffle=True,
                         random_state=42, tol=0.0001, verbose=False, warm_start=False, nesterovs_momentum=True,
                         early_stopping=True,beta_1=0.99, beta_2=0.99, epsilon=1e-5)
model_mlp.fit(X_1,y_train)
0.0648051507582861
0.0009564387736001718
0.10763974138293898

model_mlp= MLPRegressor(hidden_layer_sizes=(80,40,20),activation='relu',solver='lbfgs', alpha=0.001, batch_size=80,
                         learning_rate='constant', power_t=0.01, max_iter=10000, shuffle=True,
                         random_state=42, tol=0.001, verbose=False, warm_start=False, nesterovs_momentum=True,
                         early_stopping=True,beta_1=0.99, beta_2=0.99, epsilon=1e-5)

0.06253997160702265
-0.003297842557377432
0.10987863306706257

MLPRegressor(hidden_layer_sizes=(120,80,40,20),activation='relu',solver='lbfgs', alpha=0.001, batch_size=80,
                         learning_rate='constant', power_t=0.01, max_iter=10000, shuffle=True,
                         random_state=42, tol=0.001, verbose=False, warm_start=False, nesterovs_momentum=True,
                         early_stopping=True,beta_1=0.99, beta_2=0.99, epsilon=1e-5)

0.06195968240343149
-0.0021760710993874756
0.11161683612803143


# In[ ]:





# In[ ]:





# In[11]:


import seaborn as sns
plt.figure(figsize=(5, 5))

h = sns.jointplot(y_val,y_pred,kind = 'reg')
#'scatter', 'reg', 'resid', 'kde', or 'hex'
#h.plot_joint(sns.kdeplot, zorder=0, n_levels=6,color='red')
# JointGrid has a convenience function
h.set_axis_labels('x', 'y', fontsize=15)
plt.grid(True)
my_x_ticks = np.arange(-2.5, 1, 0.5)
my_y_ticks = np.arange(-2.5, 1, 0.5)
plt.xticks(my_x_ticks,fontsize=15)
plt.yticks(my_y_ticks,fontsize=15)
# or set labels via the axes objects
h.ax_joint.set_xlabel('$[\mathrm{Fe} / \mathrm{H}]_{\mathrm{APOGEE}}$ (dex)',fontsize = 15)
h.ax_joint.set_ylabel('$[\mathrm{Fe} / \mathrm{H}]_{\mathrm{LASSO-MLP}}$ (dex)',fontsize = 15)
plt.tight_layout()
plt.savefig('2_6_feh_线性_apogee.pdf')


# In[12]:


x = y_pred - y_val
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import matplotlib.mlab as mlab
from scipy.stats import norm
 
 

#n, bins, patches = plt.hist(x, 20, density=1, facecolor='blue', alpha=0.75)  #第二个参数是直方图柱子的数量
mu =np.mean(x) #计算均值 
sigma =np.std(x) 
num_bins = 300 #直方图柱子的数量 
n, bins, patches = plt.hist(x, num_bins,density=1) 
#直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象 
y = norm.pdf(bins, mu, sigma)#拟合一条最佳正态分布曲线y 
plt.xticks(my_x_ticks,fontsize=15)
plt.yticks(my_y_ticks,fontsize=15)
#plt.grid(True)
plt.plot(bins, y, 'r--',label = '$\mu = $'+ str(round(mu,5)) + "\n"+ '$\sigma= $'+str(round(sigma,4)))
plt.xlabel('Delta  $[\mathrm{Fe} / \mathrm{H}]$ (dex)',fontsize=15) 
plt.ylabel('Frequency',fontsize=15) #绘制y轴 
#plt.title('Histogram : $\mu$=' + str(round(mu,4)) + ' $\sigma=$'+str(round(sigma,3)))  #中文标题 u'xxx' 
#plt.subplots_adjust(left=0.15)#左边距 
#plt.savefig('20-30TeffA-P高斯.pdf')
plt.legend(loc='upper left',frameon=False,fontsize = 15)
plt.savefig('2_6_feh_高斯_apogee.eps')


# In[13]:


import seaborn as sns
feh_error = y_pred - y_val
plt.figure(figsize=(10, 5))
plt.scatter(y_val,feh_error)
plt.xlabel('$[\mathrm{Fe} / \mathrm{H}]^{\mathrm{APOGEE}}$ (dex)',fontsize=15)
plt.ylabel('Delta $[\mathrm{Fe} / \mathrm{H}]$ (dex)',fontsize=15)

plt.axhline(y=0, color='r', linestyle='-')
#plt.tick_params(top=True,bottom=False,left=False,right=False)
#plt.tick_params(labeltop=True,labelleft=True,labelright=True,labelbottom=True)

plt.savefig('2_6_feh_apogee_error.eps')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[66]:


#X_train0, X_test0, y_train0, y_test0 = train_test_split(c0, label, test_size=0.2, random_state=42)
X_train, X_val_1, y_train, y_val_1 = train_test_split(c2,train_labe3, test_size=0.2, random_state=42)
X_val = np.load('../拟合方法/fit_多项式_贫金属星.npy')
feh = [-2.79,-2.43,-2.74,-2.33,-2.48,-1.67,-2.18,-2.4,-2.8,-2.44,-2.14,-2.68,-2.63]


# In[67]:


from sklearn import svm
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

sc=StandardScaler()
sc.fit(X_train)#计算样本的均值和标准差
X_train_std=sc.transform(X_train)
X_val_std=sc.transform(X_val)

from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV 
clf = LassoCV(alphas=[0.001,0.002,0.003,0.005,0.0005,0.0008,0.01],max_iter=10000)
# clf = Lasso(alpha = 0.001, fit_intercept=True, normalize=False, precompute=True, 
#             copy_X=True, max_iter=10000, tol=1e-4, warm_start=False, 
#             positive=False, random_state=None
#            )
#alpha = 0.01

clf.fit(X_train_std,y_train)
print(clf.alpha_)
mask = clf.coef_ != 0
X_1 = X_train_std[:,mask]
#X_1 = X_train_std
print(X_1.shape)
X_2 = X_val_std[:,mask]
#X_2 = X_val_std


model_mlp= MLPRegressor(hidden_layer_sizes=(40,20,10),activation='relu',solver='lbfgs', alpha=0.001, batch_size=200,
                         learning_rate='constant', power_t=0.01, max_iter=2000, shuffle=True,
                         random_state=1, tol=0.001, verbose=False, warm_start=False, nesterovs_momentum=True,
                         early_stopping=True,beta_1=0.999, beta_2=0.999, epsilon=1e-7)
model_mlp.fit(X_1,y_train)
y_pred=model_mlp.predict(X_2)
#print(clf.alpha_)
print('MAE feh:', mean_absolute_error(y_pred,y_val))


# In[57]:


np.save('LASSO_MLP_feh_error.npy',y_val - y_pred)
np.save('LASSO_MLP_feh_pred.npy',y_pred)
np.save('LASSO_MLP_feh_val.npy',y_val)


# In[58]:


x1 = y_pred - y_val
mu =np.mean(x1) #计算均值 
sigma =np.std(x1) 
print(mu)
print(sigma)


# In[80]:


plt.figure(figsize=(10,5))
plt.plot(wavelength_new,clf.coef_)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Normalized Flux', fontsize=16)
plt.savefig('9_7feh特征提取位置.eps')


# In[41]:


clf.coef_[clf.coef_ != 0] = 1
clf.coef_


# In[42]:


plt.figure(figsize=(10,5))
plt.plot(wavelength_new,clf.coef_*X_train_std[1160])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Normalized Flux', fontsize=16)
plt.savefig('feh_max特征提取后.eps')


# In[43]:


plt.figure(figsize=(10,5))
plt.plot(wavelength_new,clf.coef_*X_train_std[525])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Normalized Flux', fontsize=16)
plt.savefig('feh_min特征提取后.eps')


# In[83]:


plt.figure(figsize=(10,5))
plt.plot(wavelength_new,X_train_std[0])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Normalized Flux', fontsize=16)
plt.savefig('9_7feh特征提取前.eps')


# In[63]:


import seaborn as sns
plt.figure(figsize=(5, 5))

h = sns.jointplot(y_val,y_pred,kind = 'reg')
#'scatter', 'reg', 'resid', 'kde', or 'hex'
#h.plot_joint(sns.kdeplot, zorder=0, n_levels=6,color='red')
# JointGrid has a convenience function
h.set_axis_labels('x', 'y', fontsize=15)
plt.grid(True)
my_x_ticks = np.arange(-2.5, 1, 0.5)
my_y_ticks = np.arange(-2.5, 1, 0.5)
plt.xticks(my_x_ticks,fontsize=15)
plt.yticks(my_y_ticks,fontsize=15)
# or set labels via the axes objects
h.ax_joint.set_xlabel('$[\mathrm{Fe} / \mathrm{H}]_{\mathrm{APOGEE}}$ (dex)',fontsize = 15)
h.ax_joint.set_ylabel('$[\mathrm{Fe} / \mathrm{H}]_{\mathrm{LASSO-MLP}}$ (dex)',fontsize = 15)
plt.tight_layout()
plt.savefig('9_7feh_线性_apogee.pdf')


# In[64]:


x = y_val - y_pred
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import matplotlib.mlab as mlab
from scipy.stats import norm
 
 

#n, bins, patches = plt.hist(x, 20, density=1, facecolor='blue', alpha=0.75)  #第二个参数是直方图柱子的数量
mu =np.mean(x) #计算均值 
sigma =np.std(x) 
num_bins = 300 #直方图柱子的数量 
n, bins, patches = plt.hist(x, num_bins,density=1) 
#直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象 
y = norm.pdf(bins, mu, sigma)#拟合一条最佳正态分布曲线y 
plt.xticks(my_x_ticks,fontsize=15)
plt.yticks(my_y_ticks,fontsize=15)
#plt.grid(True)
plt.plot(bins, y, 'r--',label = '$\mu = $'+ str(round(mu,5)) + "\n"+ '$\sigma= $'+str(round(sigma,4)))
plt.xlabel('Delta  $[\mathrm{Fe} / \mathrm{H}]$ (dex)',fontsize=15) 
plt.ylabel('Frequency',fontsize=15) #绘制y轴 
#plt.title('Histogram : $\mu$=' + str(round(mu,4)) + ' $\sigma=$'+str(round(sigma,3)))  #中文标题 u'xxx' 
#plt.subplots_adjust(left=0.15)#左边距 
#plt.savefig('20-30TeffA-P高斯.pdf')
plt.legend(loc='upper left',frameon=False,fontsize = 15)
plt.savefig('9_7feh_高斯_apogee.eps')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




