import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

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

# Add a regression plane
model = LinearRegression()
model.fit(df[['HRPre', 'EFPre', 'AdjPreLVSize']], marker)
coef = model.coef_
intercept = model.intercept_

xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(y.min(), y.max(), 10))
zz = -(coef[0] * xx + coef[1] * yy + intercept) / coef[2]
axs[0].plot_surface(xx, yy, zz, alpha=0.5)

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

axs[1].set_xlabel('HR2W')
axs[1].set_ylabel('EF2W')
axs[1].set_zlabel('AdjLVSize2W')
axs[1].set_title('Treatment vs Mortality18M vs Adjusted LV Size After 2 Weeks')

x = df['HR4W']
y = df['EF4W']
z = df['AdjLVSize4W']
c = np.where(df['Treatment']==1,'b','r')
s = df['QoL2W']
marker = df['Mortality18M']
for m in np.unique(marker):
    axs[2].scatter(x[marker==m], y[marker==m], z[marker==m], c=c[marker==m], s=s[marker==m], marker = 'o' if m==0 else 'P')

axs[2].set_xlabel('HRDelta')
axs[2].set_ylabel('EFDelta')
axs[2].set_zlabel('AdjDeltaLVSize')
axs[2].set_title('Treatment vs Mortality18M')

plt.show()
