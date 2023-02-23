import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

df = pd.read_csv('data/vis_data.csv')

# 3D plot

# set axes variables
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Compute the difference between post-treatment and pre-treatment values
dx = df['HR2W'] - df['HRPre']
dy = df['EF2W'] - df['EFPre']
dz = df['AdjLVSize2W'] - df['AdjPreLVSize']

# Color the points based on treatment
colors = np.where(df['Treatment'] == 1, 'blue', 'red')

# Create the scatter plot
ax.scatter(dx, dy, dz, c=colors, marker='o')

# Set axis labels
ax.set_xlabel('Change in HR')
ax.set_ylabel('Change in EF')
ax.set_zlabel('Change in Adjusted LV Size')

# Set axis limits
ax.set_xlim(-30, 30)
ax.set_ylim(-.2, .2)
ax.set_zlim(-1.5, 1.5)

# Set the aspect ratio of the plot to be equal
ax.set_box_aspect([1, 1, 1])

# Add a title
ax.set_title('Changes in HR, EF, and Adjusted LV Size After Treatment')

plt.show()

