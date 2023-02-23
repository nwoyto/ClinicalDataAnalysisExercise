import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import scikit_posthocs as sp
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/vis_data.csv')

fig, axs = plt.subplots(1, 3, figsize=(15,5), subplot_kw={'projection': '3d'})

x = df['HRPre']
y = df['EFPre']
z = df['AdjPreLVSize']
c = np.where(df['Treatment']==1,'b','r')
s = df['QoLPre']
marker = df['Mortality18M']
for m in np.unique(marker):
    axs[0].scatter(x[marker==m], y[marker==m], z[marker==m], c=c[marker==m], s=s[marker==m], marker = 'o' if m==0 else 'P')


axs[0].set_xlabel('HRPre')
axs[0].set_ylabel('EFPre')
axs[0].set_zlabel('AdjPreLVSize')
axs[0].set_title('Pretreatment')

x = df['HR2W']
y = df['EF2W']
z = df['AdjLVSize2W']
c = np.where(df['Treatment']==1,'b','r')
s = df['QoL2W']
marker = df['Mortality18M']
for m in np.unique(marker):
    axs[1].scatter(x[marker==m], y[marker==m], z[marker==m], c=c[marker==m], s=s[marker==m], marker = 'o' if m==0 else 'P')


axs[1].set_xlabel('HRPost')
axs[1].set_ylabel('EFPost')
axs[1].set_zlabel('AdjPostLVSize')
axs[1].set_title('2 Weeks')

x = df['HR4W']
y = df['EF4W']
z = df['AdjLVSize4W']
c = np.where(df['Treatment']==1,'b','r')
s = df['QoL4W']
marker = df['Mortality18M']
for m in np.unique(marker):
    axs[2].scatter(x[marker==m], y[marker==m], z[marker==m], c=c[marker==m], s=s[marker==m], marker = 'o' if m==0 else 'P')

axs[2].set_xlabel('HR4W')
axs[2].set_ylabel('EF4W')
axs[2].set_zlabel('AdjLVSize4W')
axs[2].set_title('4 Weeks')

plt.show()