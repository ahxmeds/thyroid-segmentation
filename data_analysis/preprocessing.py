#%%
import os 
import SimpleITK as sitk 
import shutil 
import pydicom 
from glob import glob 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed
# %%
def read_nifti_image(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).T

def read_dicom_image(path):
    dc = pydicom.dcmread(path)
    return dc.pixel_array, dc
#%%
dicomdir = '/mnt/d/Projects/Thyroid_data/DICOM'
patientdirs = os.listdir(dicomdir)
# patientdirs = [
#     'C_002932',
#     'C_002947',
#     'C_003295',
#     'C_003320',
#     'I_410723101',
#     'I_410730105',
#     'I_410828106',
#     'I_410909107',
#     'I_420225104',
#     'I_420320111',
#     'I_420407101',
#     'I_420421106',
#     'I_420424107',
#     'I_420525108',
#     'I_420901104'
# ]
stpaths = [os.path.join(dicomdir, d, f'{d}.dcm') for d in patientdirs]
gtpaths = [os.path.join(dicomdir, d, f'Untitled.nii.gz') for d in patientdirs]
# %%
# plot and watch images and masks
# %%
savedir = '/home/shadab/Projects/thyroid-segmentation/data_analysis/dicom_figures'

#%%
patientid_list, sex_list = [], []
stsizeX_list, stsizeY_list, stsizeZ_list = [],[],[]
stspacingX_list, stspacingY_list, stspacingZ_list = [],[],[]
gtsizeX_list, gtsizeY_list, gtsizeZ_list = [], [], []
gtspacingX_list, gtspacingY_list, gtspacingZ_list = [],[],[]

# def process_paths(stpath, gtpath):
for stpath, gtpath in zip(stpaths, gtpaths):
    fname = os.path.basename(os.path.dirname(gtpath))
    print(fname)
    patientid_list.append(fname)
    
    st, dc = read_dicom_image(stpath)
    sex_list.append(dc.PatientSex)
    stspacingX_list.append(dc['PixelSpacing'][0])
    stspacingY_list.append(dc['PixelSpacing'][1])
    stspacingZ_list.append('null')

    if st.ndim == 2:
        stsizeX_list.append(st.shape[0])
        stsizeY_list.append(st.shape[1])
        stsizeZ_list.append('null')
    
    elif st.ndim == 3:
        stsizeX_list.append(st.shape[1])
        stsizeY_list.append(st.shape[2])
        stsizeZ_list.append(st.shape[0])
    else:
        pass
    

    gtimg = sitk.ReadImage(gtpath)
    gtsizeX_list.append(gtimg.GetSize()[0])
    gtsizeY_list.append(gtimg.GetSize()[1])
    gtsizeZ_list.append(gtimg.GetSize()[2])
    gtspacingX_list.append(gtimg.GetSpacing()[0])
    gtspacingY_list.append(gtimg.GetSpacing()[1])
    gtspacingZ_list.append(gtimg.GetSpacing()[2])


# %%
colnames = [
    'PatientID',
    'Sex',
    'STSizeX',
    'STSizeY',
    'STSizeZ',
    'GTSizeX',
    'GTSizeY',
    'GTSizeZ',
    'STSpacingX',
    'STSpacingY',
    'STSpacingZ',
    'GTSpacingX',
    'GTSpacingY',
    'GTSpacingZ'
]
data_info = pd.DataFrame(columns=colnames)

data_info['PatientID'] = patientid_list
data_info['Sex'] = sex_list
data_info['STSizeX'] = stsizeX_list
data_info['STSizeY'] = stsizeY_list
data_info['STSizeZ'] = stsizeZ_list
data_info['GTSizeX'] = gtsizeX_list
data_info['GTSizeY'] = gtsizeY_list
data_info['GTSizeZ'] = gtsizeZ_list
data_info['STSpacingX'] = stspacingX_list
data_info['STSpacingY'] = stspacingY_list
data_info['STSpacingZ'] = stspacingZ_list
data_info['GTSpacingX'] = gtspacingX_list
data_info['GTSpacingY'] = gtspacingY_list
data_info['GTSpacingZ'] = gtspacingZ_list
data_info.to_csv('datainfo_version01.csv', index=False)


    # gt = read_nifti_image(gtpath)

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(st)
    # ax[1].imshow(gt[:,:,0].T)
    # savepath = os.path.join(savedir, f'{fname}.png')
    # fig.savefig(savepath, dpi=120, bbox_inches='tight')
    # plt.close(fig)

    # except: 
    #     failed_cases.append(fname)

# Parallel(n_jobs=4)(delayed(process_paths)(stpath, gtpath) for stpath, gtpath in zip(stpaths, gtpaths))

# %%
images_dir = '/mnt/d/Projects/Thyroid_data/NIFTI/images'
labels_dir = '/mnt/d/Projects/Thyroid_data/NIFTI/labels'
for index, row in data_info.iterrows():
    ptid = row['PatientID']
    stpath = os.path.join(dicomdir, f'{ptid}', f'{ptid}.dcm')
    gtpath = os.path.join(dicomdir, f'{ptid}', f'Untitled.nii.gz')

    st = pydicom.dcmread(stpath).pixel_array 

    if st.ndim == 2:
        st_final = st 
    elif st.ndim == 3:
        st_final = st[0]
    else:
        pass
    
    gtimg = sitk.ReadImage(gtpath)
    gt = read_nifti_image(gtpath)
    if gt.shape[2] == 1:
        gt_final = gt[:,:,0].T
    elif gt.shape[2] == 2:
        gt_final = gt[:,:,1].T
    else:
        pass

    stimg_final = sitk.GetImageFromArray(st_final.T)
    stimg_final.SetSpacing(gtimg.GetSpacing()[:-1])
    stimg_final.SetOrigin(gtimg.GetOrigin()[:-1])
    stimg_final.SetDirection(gtimg.GetDirection()[:-1])

    gtimg_final = sitk.GetImageFromArray(gt_final.T)
    gtimg_final.SetSpacing(gtimg.GetSpacing()[:-1])
    gtimg_final.SetOrigin(gtimg.GetOrigin()[:-1])
    gtimg_final.SetDirection(gtimg.GetDirection()[:-1])

    stsavepath = os.path.join(images_dir, f'{ptid}.nii.gz')
    gtsavepath = os.path.join(labels_dir, f'{ptid}.nii.gz')

    sitk.WriteImage(stimg_final, stsavepath)
    gitk.WriteImage(gtimg_final, gtsavepath)

    print(f'Done with {index}: {ptid}')