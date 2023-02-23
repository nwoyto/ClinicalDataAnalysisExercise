import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

# Load data from CSV file
df = pd.read_csv('RegeneronProject/data/vis_data.csv')

# Create 3D figure with subplots
fig, axs = plt.subplots(1, 3, figsize=(15,5), subplot_kw={'projection': '3d'})

# Plot the first subplot
x = df['HRPre']
y = df['EFPre']
z = df['AdjPreLVSize']
c = np.where(df['Treatment']==1,'b','r')
s = df['QoLPre']
marker = df['Mortality18M']
for m in np.unique(marker):
    axs[0].scatter(x[marker==m], y[marker==m], z[marker==m], c=c[marker==m], s=s[marker==m], marker = 'o' if m==0 else 'P')

# Add a regression plane to the first subplot
model = LinearRegression()
model.fit(df[['HRPre', 'EFPre', 'AdjPreLVSize']], marker)
coef = model.coef_
intercept = model.intercept_

xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(y.min(), y.max(), 10))
zz = -(coef[0] * xx + coef[1] * yy + intercept) / coef[2]
axs[0].plot_surface(xx, yy, zz, alpha=0.5)

# Set labels and title for the first subplot
axs[0].set_xlabel('HRPre')
axs[0].set_ylabel('EFPre')
axs[0].set_zlabel('AdjPreLVSize')
axs[0].set_title('Pretreatment')

# Plot the second subplot
x = df['HR2W']
y = df['EF2W']
z = df['AdjLVSize2W']
c = np.where(df['Treatment']==1,'b','r')
s = df['QoL2W']
marker = df['Mortality18M']
for m in np.unique(marker):
    axs[1].scatter(x[marker==m], y[marker==m], z[marker==m], c=c[marker==m], s=s[marker==m], marker = 'o' if m==0 else 'P')

# Add a regression plane to the second subplot
model = LinearRegression()
model.fit(df[['HR2W', 'EF2W', 'AdjLVSize2W']], marker)
coef = model.coef_
intercept = model.intercept_

xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(y.min(), y.max(), 10))
zz = -(coef[0] * xx + coef[1] * yy + intercept) / coef[2]
axs[1].plot_surface(xx, yy, zz, alpha=0.5)

# Set labels and title for the second subplot
axs[1].set_xlabel('HR2W')
axs[1].set_ylabel('EF2W')
axs[1].set_zlabel('AdjLVSize2W')
axs[1].set_title('2 Weeks')

# Plot the third subplot
x = df['HR4W']
y = df['EF4W']
z = df['AdjLVSize4W']
c = np.where(df['Treatment']==1,'b','r')
s = df['QoL2W']
marker = df['Mortality18M']
for m in np.unique(marker):
    axs[2].scatter(x[marker==m], y[marker==m], z[marker==m], c=c[marker==m], s=s[marker==m], marker = 'o' if m==0 else 'P')

# Add a regression plane
model = LinearRegression()
model.fit(df[['HR4W', 'EF4W', 'AdjLVSize4W']], marker)
coef = model.coef_
intercept = model.intercept_

xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(y.min(), y.max(), 10))
zz = -(coef[0] * xx + coef[1] * yy + intercept) / coef[2]
axs[2].plot_surface(xx, yy, zz, alpha=0.5)

axs[2].set_xlabel('HR4W')
axs[2].set_ylabel('EF4W')
axs[2].set_zlabel('AdjLVSize4W')
axs[2].set_title('4 Weeks')


plt.show()