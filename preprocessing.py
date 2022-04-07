#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:



data = pd.read_csv("../DR820_30_唯一交叉匹配.csv")
#data = pd.read_csv("../feh训练集3sigma外7111条.csv")


# In[3]:


data


# In[4]:


flux_list1 = []
wavelengths_list1 = []
flux_list = []
wavelengths_list = []
min_max = []
flux_interp1d_list = []

valid_index0 = np.where(np.nan_to_num(data['combined_z']) != -9999.99)[0]
param1 = data.iloc[valid_index0]
valid_index = np.where(param1['combined_z'] != -9999)[0] 
valid_param = param1.iloc[valid_index]


# In[5]:


valid_param


# In[6]:


# parameters = pd.read_csv('fits/valid_param.csv', sep='|')
for _, p in valid_param.iterrows(): 
    fname = p['combined_file']
    fname = fname.split('/', 2)[-1]
    # read fits
    #hdu = fits.open('../高信噪比交叉匹配fits/' + fname)
    hdu = fits.open('../DR820_30交叉匹配fits/' + fname)
    #hdu = fits.open('../LAMOST_DR8_贫金属星fits/' + fname)
    flux_list1.append(hdu[0].data[0])
    wavelengths_list1.append(hdu[0].data[2])
print(np.shape(flux_list1))
print(np.shape(wavelengths_list1))


# In[7]:


# remove red_shift and find the wavelength of the common interval
wavelength4 = []
for i in range(len(wavelengths_list1)):
    wavelength1 = wavelengths_list1[i]
    wavelength2 = np.log10(wavelength1) - np.log10(1 + valid_param['combined_z'][valid_index[i]])
    wavelength3 = 10 ** (wavelength2)
    wavelength4.append(wavelength3)


# In[8]:


wavelength_new = np.load('./wavelength_new_un.npy')
print(wavelength_new)


# In[9]:


np.shape(wavelength_new)


# In[10]:


flux_interp1d_list=[]
for i in range(len(wavelength4)):
    wavelength = wavelength4[i]
    wavelength = np.log10(wavelength) - np.log10(1 + valid_param['combined_z'][valid_index[i]])
    wavelength = 10 ** (wavelength)
    flux = flux_list1[i]
    f = interp1d(wavelength, flux, fill_value="extrapolate") #内插值
    flux_new = f(wavelength_new)
    flux_interp1d_list.append(flux_new)
print(np.shape(flux_interp1d_list))


# In[11]:


np.save('./flux_interp.npy',flux_interp1d_list)

import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve


# In[12]:


flux_interp = np.load('./flux_interp.npy')


# In[13]:


def csp_polyfit(sp,angs,param):
           
    # standardize flux
    sp_c = np.mean(sp)                           #　流量中值
    sp = sp - sp_c                                #　流量中心化
    sp_s = np.std(sp)   #　流量均方差
    sp = sp / sp_s
    
    # standardize wavelength
    angs_c = np.mean(angs)
    angs = angs - angs_c
    angs_s = np.std(angs)
    angs = angs/angs_s
    
    param['poly_sp_c'] = sp_c
    param['poly_sp_s'] = sp_s
    param['poly_angs_c'] = angs_c
    param['poly_angs_s'] = angs_s
    
    data_flag = np.full(sp.shape, 1)
    
    i = 0
    con = True
    while(con):
        P_g = np.polyfit(angs, sp, param['poly_global_order'])  # coefficient
        param['poly_P_g'] = P_g
        fitval_1 = np.polyval(P_g, angs)   # 拟合y值
        dev = fitval_1 - sp
        sig_g = np.std(dev)
        
        data_flag_new = (dev > (-param['poly_upperlimit'] * sig_g)) * (dev < (param['poly_lowerlimit'] * sig_g))
    
        if sum(abs( data_flag_new - data_flag ))>0:
            if param['poly_del_filled'] == 1: 
                data_flag = data_flag_new
            else:
                fill_flag = data_flag - data_flag_new
                index_1 = np.where(fill_flag != 0)
                sp[index_1] = fitval_1[index_1]
        else:
            con = False
        i += 1
    
    index_2 = np.where(data_flag != 0)
    param['poly_sp_filtered'] = sp[index_2]
    param['poly_angs_filtered'] = angs[index_2]
    
    return param


# In[14]:


def sp_median_polyfit1stage(flux,lambda_log,param):
    flux1 = flux
    lambda1 = lambda_log

# Median filter
    flux_median1 = medfilt(flux1, param['median_radius']) 

#用中值滤波结果预剔除 
    dev1 = flux_median1 - flux1
    sigma = np.std(dev1)
    data_flag1 = (dev1 < (param['poly_lowerlimit'] * sigma)) * (dev1 > (-param['poly_upperlimit'] * sigma))
    
    # 用中值滤波结果预剔除伪谱线
    fill_flag1 = 1 - data_flag1
    
    if param['poly_del_filled'] == 1:
        index_1 = np.where(data_flag1)
        flux1 = flux1[index_1]
        lambda1 = lambda1[index_1]
    elif param['poly_del_filled'] == 2:
        index_2 = np.where(fill_flag1)
        flux1[index_2] = flux_median1[index_2]
#迭代拟合连续谱
    param = csp_polyfit(flux1, lambda1, param)
# 计算原始采样点处的连续谱样本
#  波长预处理
    angs = lambda1 - param['poly_angs_c']
    angs = angs / param['poly_angs_s']
# 连续谱样本
    fitval_g = np.polyval(param['poly_P_g'], angs)
    continum_fitted = fitval_g * param['poly_sp_s'] + param['poly_sp_c']
    if param['poly_SM'] ==1: 
        angss = lambda1
    else: 
        angss = 10 ** lambda1
    #print("--------------------------- -------------------")
    ##plt.figure(figsize=(16,9))
    #plt.plot(angss, flux2, 'r--', label = 'linear')
    #plt.plot(angss, continum_fitted)
    #plt.show()
    #print("----------------------------------------------")   
    return continum_fitted


# In[15]:


# import numpy as np 
# #import cv2
# flux_end_train = []
# wavelength_new = np.log10(wavelength_new)
# for i in tqdm(range(len(flux_interp1d_list))):
#     param = {'poly_global_order':8,'nor':1,'poly_lowerlimit':7,'poly_upperlimit':1, 'median_radius':5,'poly_SM':0,'poly_del_filled':3} 
#     flux2 = flux_interp1d_list[i]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
#     #continum_fitted = cv2.blur(flux2,wavelength_new, param)
#     continum_fitted = sp_median_polyfit1stage(flux2,wavelength_new, param)
#     flux_end = flux2 / continum_fitted
#     flux_end_train.append(flux_end)


import numpy as np 
#import cv2
flux_end_train = []
continum_fitted_list = []
wavelength_new = np.log10(wavelength_new)
for i in tqdm(range(len(flux_interp1d_list))):
    param = {'poly_global_order':6,'nor':1,'poly_lowerlimit':3,'poly_upperlimit':5, 'median_radius':3,'poly_SM':0,'poly_del_filled':2}
    flux2 = flux_interp1d_list[i]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    #continum_fitted = cv2.blur(flux2,wavelength_new, param)
    continum_fitted = sp_median_polyfit1stage(flux2,wavelength_new, param)
    flux_end = flux2 / continum_fitted
    flux_end_train.append(flux_end)
    continum_fitted_list.append(continum_fitted)


# In[16]:


np.save('./fit_多项式_low_new.npy', flux_end_train)


# In[17]:


#c0 = np.load('../flux_fiited0.npy')


c2 = np.load('./fit_多项式_low_new.npy')
#flux_interp = np.load('./flux_interp.npy')


wavelength_new = np.load('./wavelength_new_un.npy')


# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

i=0

f2 = flux_interp[i]/ c2[i]
plt.figure(figsize=(8,6))
#[3839.52825828 3840.41244413 3841.29683358 ... 8932.59961878 8934.65666267
# 8936.71418026]
print(np.shape(flux_list1))
print(np.shape(wavelengths_list1))

plt.subplot(2,1,1)
plt.plot(wavelengths_list1[i], flux_list1[i],'k-')
plt.xlim([wavelengths_list1[i].min(), wavelengths_list1[i].max()])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Original Flux', fontsize=16)
plt.subplot(2,1,2)
#plt.plot(wavelength_new,c0[i],'k')
# plt.plot(wavelength_new,c1[i],'b')
plt.plot(wavelength_new,flux_end_train[i],'m')
# plt.plot(wavelength_new,c3[i],'m')
plt.axhline(y=1,color='r', linestyle='--')
plt.xlim([wavelengths_list1[i].min(), wavelengths_list1[i].max()])
#plt.xlim([wavelength_new.min(), wavelength_new.max()])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Normalized Flux', fontsize=16)

plt.tight_layout()
plt.savefig('1_12_预处理前后.eps')


# In[19]:


teff_val = np.load('../参数估计方法_多项式/teff_val.npy')
teff_pred = np.load('../参数估计方法_多项式/teff_pred.npy')
teff_error = np.load('../参数估计方法_多项式/teff_error.npy')

logg_val = np.load('../参数估计方法_多项式/logg_val.npy')
logg_pred = np.load('../参数估计方法_多项式/logg_pred.npy')
logg_error = np.load('../参数估计方法_多项式/logg_error.npy')

feh_val = np.load('../参数估计方法_多项式/feh_val.npy')
feh_pred = np.load('../参数估计方法_多项式/feh_pred.npy')
feh_error = np.load('../参数估计方法_多项式/feh_error.npy')


# In[20]:


data.iloc[2288]


# In[21]:


teff_pred[964]


# In[22]:


data.iloc[2288]['combined_file']


# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

i=2288

f2 = flux_interp[i]/ c2[i]
plt.figure(figsize=(10,4))

fname = '20130226/GAC089N28V1/spec-56350-GAC089N28V1_sp12-034.fits.gz'
fname = fname.split('/', 2)[-1]
    # read fits
    #hdu = fits.open('../高信噪比交叉匹配fits/' + fname)
hdu = fits.open('../DR820_30交叉匹配fits/' + fname)
flux = hdu[0].data[0]
wavelengths = hdu[0].data[2]


# plt.subplot(3,1,1)
# plt.plot(wavelengths,flux,'k-')
# plt.xlim([wavelengths.min(), wavelengths.max()])
# plt.ylim([flux.min(), flux.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.vlines(3839.52825828,flux.min(),flux.max(),colors = "r",linestyles = "dashed")
# plt.vlines(8936.71418026,flux.min(), flux.max(),colors = "r",linestyles = "dashed")
# plt.xlabel('Wavelength($\mathrm{\AA}$)')
# plt.ylabel('Original Flux')

#plt.subplot(3,1,2)
plt.plot(wavelength_new, flux_interp[i],'k-')
plt.plot(wavelength_new,continum_fitted_list[i],'b-')
plt.xlim([wavelength_new.min(), wavelength_new.max()])
plt.ylim([flux.min(), flux.max()])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Original Flux', fontsize=16)

# plt.subplot(3,1,3)
# #plt.plot(wavelength_new,c0[i],'k')
# # plt.plot(wavelength_new,c1[i],'b')
# plt.plot(wavelength_new,c2[i],'m')
# # plt.plot(wavelength_new,c3[i],'m')
# plt.axhline(y=1,color='r', linestyle='--')
# plt.xlim([wavelength_new.min(), wavelength_new.max()])
# plt.ylim([-1, 100])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
# plt.ylabel('Normalized Flux', fontsize=16)

#plt.tight_layout()
#plt.savefig('1_12_teff_max.eps')

#流量存在0
plt.tight_layout()
plt.savefig('3_14_teff_max.eps')


# In[24]:


data.iloc[0]


# In[25]:


teff_pred[1471]


# In[26]:


data.iloc[0]['combined_file']


# In[27]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

i=0
f2 = flux_interp[i]/ c2[i]
plt.figure(figsize=(10,4))

fname = '20111024/M5901/spec-55859-M5901_sp03-132.fits.gz'
fname = fname.split('/', 2)[-1]
    # read fits
    #hdu = fits.open('../高信噪比交叉匹配fits/' + fname)
hdu = fits.open('../DR820_30交叉匹配fits/' + fname)
flux = hdu[0].data[0]
wavelengths = hdu[0].data[2]


# plt.subplot(3,1,1)
# plt.plot(wavelengths,flux,'k-')
# plt.xlim([wavelengths.min(), wavelengths.max()])
# plt.ylim([flux.min(), flux.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.vlines(3839.52825828,flux.min(),flux.max(),colors = "r",linestyles = "dashed")
# plt.vlines(8936.71418026,flux.min(), flux.max(),colors = "r",linestyles = "dashed")
# plt.xlabel('Wavelength($\mathrm{\AA}$)')
# plt.ylabel('Original Flux')

#plt.subplot(3,1,2)
plt.plot(wavelength_new, flux_interp[i],'k-')
plt.plot(wavelength_new,continum_fitted_list[i],'b-')
plt.xlim([wavelength_new.min(), wavelength_new.max()])
plt.ylim([flux.min(), flux.max()])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Original Flux', fontsize=16)

# plt.subplot(3,1,3)
# #plt.plot(wavelength_new,c0[i],'k')
# # plt.plot(wavelength_new,c1[i],'b')
# plt.plot(wavelength_new,c2[i],'m')
# # plt.plot(wavelength_new,c3[i],'m')
# plt.axhline(y=1,color='r', linestyle='--')
# plt.xlim([wavelength_new.min(), wavelength_new.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
# plt.ylabel('Normalized Flux', fontsize=16)

plt.tight_layout()
plt.savefig('3_14_teff_min.eps')


# In[28]:


data.iloc[3027]


# In[29]:


logg_pred[1109]


# In[30]:


data.iloc[3027]['combined_file']


# In[31]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

i=3027
f2 = flux_interp[i]/ c2[i]
plt.figure(figsize=(10,4))
fname = '20130421/HD145433N463537B01/spec-56404-HD145433N463537B01_sp04-177.fits.gz'
fname = fname.split('/', 2)[-1]
    # read fits
    #hdu = fits.open('../高信噪比交叉匹配fits/' + fname)
hdu = fits.open('../DR820_30交叉匹配fits/' + fname)
flux = hdu[0].data[0]
wavelengths = hdu[0].data[2]


# plt.subplot(3,1,1)
# plt.plot(wavelengths,flux,'k-')
# plt.xlim([wavelengths.min(), wavelengths.max()])
# plt.ylim([flux.min(), flux.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.vlines(3839.52825828,flux.min(),flux.max(),colors = "r",linestyles = "dashed")
# plt.vlines(8936.71418026,flux.min(), flux.max(),colors = "r",linestyles = "dashed")
# plt.xlabel('Wavelength($\mathrm{\AA}$)')
# plt.ylabel('Original Flux')

#plt.subplot(3,1,2)
plt.plot(wavelength_new, flux_interp[i],'k-')
plt.plot(wavelength_new,continum_fitted_list[i],'b-')
plt.xlim([wavelength_new.min(), wavelength_new.max()])
plt.ylim([flux.min(), flux.max()])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Original Flux', fontsize=16)

# plt.subplot(3,1,3)
# #plt.plot(wavelength_new,c0[i],'k')
# # plt.plot(wavelength_new,c1[i],'b')
# plt.plot(wavelength_new,c2[i],'m')
# # plt.plot(wavelength_new,c3[i],'m')
# plt.axhline(y=1,color='r', linestyle='--')
# plt.xlim([wavelength_new.min(), wavelength_new.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
# plt.ylabel('Normalized Flux', fontsize=16)

plt.tight_layout()
plt.savefig('3_14_logg_max.eps')


# In[32]:


data.iloc[8756]


# In[33]:


logg_pred[1446]


# In[34]:


data.iloc[8756]['combined_file']


# In[35]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

i=8756
f2 = flux_interp[i]/ c2[i]
plt.figure(figsize=(10,4))

fname = '20170102/HD114322N280318B01/spec-57756-HD114322N280318B01_sp07-103.fits.gz'
fname = fname.split('/', 2)[-1]
    # read fits
    #hdu = fits.open('../高信噪比交叉匹配fits/' + fname)
hdu = fits.open('../DR820_30交叉匹配fits/' + fname)
flux = hdu[0].data[0]
wavelengths = hdu[0].data[2]

# plt.subplot(3,1,1)
# plt.plot(wavelengths,flux,'k-')
# plt.xlim([wavelengths.min(), wavelengths.max()])
# plt.ylim([flux.min(), flux.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.vlines(3839.52825828,flux.min(),flux.max(),colors = "r",linestyles = "dashed")
# plt.vlines(8936.71418026,flux.min(), flux.max(),colors = "r",linestyles = "dashed")
# plt.xlabel('Wavelength($\mathrm{\AA}$)')
# plt.ylabel('Original Flux')

#plt.subplot(3,1,2)
plt.plot(wavelength_new, flux_interp[i],'k-')
plt.plot(wavelength_new,continum_fitted_list[i],'b-')
plt.xlim([wavelength_new.min(), wavelength_new.max()])
plt.ylim([flux.min(), flux.max()])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Original Flux', fontsize=16)

# plt.subplot(3,1,3)
# #plt.plot(wavelength_new,c0[i],'k')
# # plt.plot(wavelength_new,c1[i],'b')
# plt.plot(wavelength_new,c2[i],'m')
# # plt.plot(wavelength_new,c3[i],'m')
# plt.axhline(y=1,color='r', linestyle='--')
# plt.xlim([wavelength_new.min(), wavelength_new.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
# plt.ylabel('Normalized Flux', fontsize=16)

plt.tight_layout()
plt.savefig('3_14_logg_min.eps')


# In[36]:


data.iloc[9736]


# In[37]:


feh_pred[671]


# In[38]:


data.iloc[9736]['combined_file']


# In[39]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

i=9736
f2 = flux_interp[i]/ c2[i]
plt.figure(figsize=(10,4))

fname = '20171115/HD080942N253332V01/spec-58073-HD080942N253332V01_sp14-024.fits.gz'
fname = fname.split('/', 2)[-1]
    # read fits
    #hdu = fits.open('../高信噪比交叉匹配fits/' + fname)
hdu = fits.open('../DR820_30交叉匹配fits/' + fname)
flux = hdu[0].data[0]
wavelengths = hdu[0].data[2]

# plt.subplot(3,1,1)
# plt.plot(wavelengths,flux,'k-')
# plt.xlim([wavelengths.min(), wavelengths.max()])
# plt.ylim([flux.min(), flux.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.vlines(3839.52825828,flux.min(),flux.max(),colors = "r",linestyles = "dashed")
# plt.vlines(8936.71418026,flux.min(), flux.max(),colors = "r",linestyles = "dashed")
# plt.xlabel('Wavelength($\mathrm{\AA}$)')
# plt.ylabel('Original Flux')

# plt.subplot(3,1,2)
plt.plot(wavelength_new, flux_interp[i],'k-')
plt.plot(wavelength_new,continum_fitted_list[i],'b-')
plt.xlim([wavelength_new.min(), wavelength_new.max()])
plt.ylim([flux.min(), flux.max()])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Original Flux', fontsize=16)

# plt.subplot(3,1,3)
# #plt.plot(wavelength_new,c0[i],'k')
# # plt.plot(wavelength_new,c1[i],'b')
# plt.plot(wavelength_new,c2[i],'m')
# # plt.plot(wavelength_new,c3[i],'m')
# plt.axhline(y=1,color='r', linestyle='--')
# plt.xlim([wavelength_new.min(), wavelength_new.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
# plt.ylabel('Normalized Flux', fontsize=16)

plt.tight_layout()
plt.savefig('3_14_feh_max.eps')


# In[40]:


data.iloc[0]


# In[41]:


feh_pred[1471]


# In[42]:


data.iloc[0]['combined_file']


# In[43]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

i=0
f2 = flux_interp[i]/ c2[i]
plt.figure(figsize=(10,4))

fname = '20111024/M5901/spec-55859-M5901_sp03-132.fits.gz'
fname = fname.split('/', 2)[-1]
    # read fits
    #hdu = fits.open('../高信噪比交叉匹配fits/' + fname)
hdu = fits.open('../DR820_30交叉匹配fits/' + fname)
flux = hdu[0].data[0]
wavelengths = hdu[0].data[2]

# plt.subplot(3,1,1)
# plt.plot(wavelengths,flux,'k-')
# plt.xlim([wavelengths.min(), wavelengths.max()])
# plt.ylim([flux.min(), flux.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.vlines(3839.52825828,flux.min(),flux.max(),colors = "r",linestyles = "dashed")
# plt.vlines(8936.71418026,flux.min(), flux.max(),colors = "r",linestyles = "dashed")
# plt.xlabel('Wavelength($\mathrm{\AA}$)')
# plt.ylabel('Original Flux')

# plt.subplot(3,1,2)
plt.plot(wavelength_new, flux_interp[i],'k-')
plt.plot(wavelength_new,continum_fitted_list[i],'b-')
plt.xlim([wavelength_new.min(), wavelength_new.max()])
plt.ylim([flux.min(), flux.max()])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Original Flux', fontsize=16)

# plt.subplot(3,1,3)
# #plt.plot(wavelength_new,c0[i],'k')
# # plt.plot(wavelength_new,c1[i],'b')
# plt.plot(wavelength_new,c2[i],'m')
# # plt.plot(wavelength_new,c3[i],'m')
# plt.axhline(y=1,color='r', linestyle='--')
# plt.xlim([wavelength_new.min(), wavelength_new.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
# plt.ylabel('Normalized Flux', fontsize=16)


plt.tight_layout()
plt.savefig('3_14_feh_min.eps')


# In[44]:


52905206
96103109
108914208
135714007
257105008
314512019
350912074
362013081
554109165
554806184
594215224
602712243
655506159


# In[19]:


h = np.where(data['combined_obsid']==108914208)[0]


# In[20]:


h


# In[21]:


data.iloc[1713]['combined_file']


# In[25]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

i=1713
f2 = flux_interp[i]/ c2[i]
plt.figure(figsize=(10,4))

fname = '20130117/GAC105N24V2/spec-56310-GAC105N24V2_sp14-208.fits.gz'
fname = fname.split('/', 2)[-1]
    # read fits
    #hdu = fits.open('../高信噪比交叉匹配fits/' + fname)
hdu = fits.open('../DR820_30交叉匹配fits/' + fname)
flux = hdu[0].data[0]
wavelengths = hdu[0].data[2]

# plt.subplot(3,1,1)
# plt.plot(wavelengths,flux,'k-')
# plt.xlim([wavelengths.min(), wavelengths.max()])
# plt.ylim([flux.min(), flux.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.vlines(3839.52825828,flux.min(),flux.max(),colors = "r",linestyles = "dashed")
# plt.vlines(8936.71418026,flux.min(), flux.max(),colors = "r",linestyles = "dashed")
# plt.xlabel('Wavelength($\mathrm{\AA}$)')
# plt.ylabel('Original Flux')

# plt.subplot(3,1,2)
plt.plot(wavelength_new, flux_interp[i],'k-')
plt.plot(wavelength_new,continum_fitted_list[i],'b-')
plt.xlim([wavelength_new.min(), wavelength_new.max()])
plt.ylim([flux.min(), flux.max()])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
plt.ylabel('Original Flux', fontsize=16)

# plt.subplot(3,1,3)
# #plt.plot(wavelength_new,c0[i],'k')
# # plt.plot(wavelength_new,c1[i],'b')
# plt.plot(wavelength_new,c2[i],'m')
# # plt.plot(wavelength_new,c3[i],'m')
# plt.axhline(y=1,color='r', linestyle='--')
# plt.xlim([wavelength_new.min(), wavelength_new.max()])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Wavelength($\mathrm{\AA}$)', fontsize=16)
# plt.ylabel('Normalized Flux', fontsize=16)


plt.tight_layout()
plt.savefig('3_14_feh_min.eps')


# In[ ]:





# In[ ]:





# In[ ]:




