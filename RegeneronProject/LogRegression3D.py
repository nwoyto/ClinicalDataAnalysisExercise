import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# Load data from CSV file
df = pd.read_csv('data/vis_data.csv')

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
# fit logistic regression model
model = LogisticRegression(fit_intercept=True)
model.fit(df[['HRPre', 'EFPre', 'AdjPreLVSize']], marker)

# create meshgrid for surface plot
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(y.min(), y.max(), 10))
zz = -(model.intercept_ + model.coef_[0][0] * xx + model.coef_[0][1] * yy) / model.coef_[0][2]
probas = model.predict_proba(np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())))
probas = probas[:, 1].reshape(xx.shape)

# add surface plot to subplot
axs[0].plot_surface(xx, yy, zz, alpha=0.5, facecolors=plt.cm.RdBu(probas))

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


# fit logistic regression model
model = LogisticRegression(fit_intercept=True)
model.fit(df[['HR2W', 'EF2W', 'AdjLVSize2W']], marker)

# create meshgrid for surface plot
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(y.min(), y.max(), 10))
zz = -(model.intercept_ + model.coef_[0][0] * xx + model.coef_[0][1] * yy) / model.coef_[0][2]
probas = model.predict_proba(np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())))
probas = probas[:, 1].reshape(xx.shape)

# add surface plot to subplot
axs[1].plot_surface(xx, yy, zz, alpha=0.5, facecolors=plt.cm.RdBu(probas))


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

# fit logistic regression model
model = LogisticRegression(fit_intercept=True)
model.fit(df[['HR4W', 'EF4W', 'AdjLVSize4W']], marker)

# create meshgrid for surface plot
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(y.min(), y.max(), 10))
zz = -(model.intercept_ + model.coef_[0][0] * xx + model.coef_[0][1] * yy) / model.coef_[0][2]
probas = model.predict_proba(np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())))
probas = probas[:, 1].reshape(xx.shape)

# add surface plot to subplot
axs[2].plot_surface(xx, yy, zz, alpha=0.5, facecolors=plt.cm.RdBu(probas))

axs[2].set_xlabel('HR4W')
axs[2].set_ylabel('EF4W')
axs[2].set_zlabel('AdjLVSize4W')
axs[2].set_title('4 Weeks')

plt.show()