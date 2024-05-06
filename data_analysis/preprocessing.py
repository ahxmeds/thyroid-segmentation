#%%
import os 
import SimpleITK as sitk 
import shutil 
import pydicom 
from glob import glob 
import numpy as np
import matplotlib.pyplot as plt 
# %%
def read_nifti_image(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).T

def read_dicom_image(path):
    return pydicom.dcmread(path).pixel_array
#%%
dicomdir = '/data/blobfuse/default/thyroid-segmentation-results/data/dicom/Data_anonymous'
patientdirs = os.listdir(dicomdir)[:-1]
stpaths = [os.path.join(dicomdir, d, f'{d}.dcm') for d in patientdirs]
gtpaths = [os.path.join(dicomdir, d, f'Untitled.nii.gz') for d in patientdirs]
# %%
# plot and watch images and masks
i = 0
stpath = stpaths[i]
gtpath = gtpaths[i]

gt = read_nifti_image(gtpath)
# %%
savedir = '/data/blobfuse/default/thyroid-segmentation-results/data/visualization/from_dicom'

for stpath, gtpath in zip(stpaths, gtpaths):
    fname = os.path.basename(os.path.dirname(gtpath))
    st = read_dicom_image(stpath)
    gt = read_nifti_image(gtpath)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(st)
    ax[1].imshow(gt[:,:,0])
    savepath = os.path.join(savedir, f'{fname}.png')
    fig.savefig(savepath, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close('all')
# %%
