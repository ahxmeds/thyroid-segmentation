#%%
import SimpleITK as sitk 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os 
from glob import glob 
import sys 
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import DATA_FOLDER, WORKING_FOLDER

def read_nifti_image(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).T
# %%
stpaths = sorted(glob(os.path.join(DATA_FOLDER, 'images', '*.nii.gz')))
gtpaths = sorted(glob(os.path.join(DATA_FOLDER, 'labels', '*.nii.gz')))
# %%
ST_max, ST_Thyroid_max = [], []
for stpath, gtpath in zip(stpaths, gtpaths):
    st = read_nifti_image(stpath)
    gt = read_nifti_image(gtpath)

    st_max = np.max(st)
    st_thyroid_max = np.max(st*gt)
    ST_max.end(st_max)
    ST_Thyroid_max.append(st_thyroid_max)
    print(f'Done {os.path.basename(gtpath)}')

# %%
fig, ax = plt.subplots(1,2)
ax[0].hist(ST_max, bins=20)
ax[1].hist(ST_Thyroid_max, bins=20)
plt.show()
# %%
ST_values, ST_Thyroid_values = [], []
for stpath, gtpath in zip(stpaths, gtpaths):
    st = read_nifti_image(stpath)
    gt = read_nifti_image(gtpath)

    st_vals = st.flatten()
    st_thyroid_vals = (st*gt).flatten()
    ST_values.extend(st_vals)
    ST_Thyroid_values.extend(st_thyroid_vals)
    print(f'Done {os.path.basename(gtpath)}')

# %%
np.save('st.npy', np.array(ST_values))
np.save('st_thyroid.npy', np.array(ST_Thyroid_values))
#%%
ST_values = np.load('st.npy')
ST_Thyroid_values = np.load('st_thyroid.npy')
#%%

fig, ax = plt.subplots()
# bins = np.logspace(np.log10(min(ST_values)), np.log10(max(ST_values)), num=50)
# ax.hist(np.array(ST_values), bins=100)
counts, bin_edges = np.histogram(ST_Thyroid_values, bins=100)  # You can adjust the number of bins
# Normalize the counts to percentage
total_counts = np.sum(counts)
frequency_percentage = (counts / total_counts) * 100
ax.hist(np.array(ST_Thyroid_values), bins=100)
ax.set_xlim(-1,2000)
plt.show()
# %%
