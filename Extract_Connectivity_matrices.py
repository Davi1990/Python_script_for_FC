#Importing packages
import numpy as np
import nilearn as nl
import nibabel as nib
import os
import glob


# Defining timeseries extraction
def extract_mask_timeseries(mask,fmri):
    """
    extract mask timeseris
    Input:
    mask: a 3D matrix with 0 and 1
    fmri: fmri 4D volume

    Return:
    a matrix with timepoints*voxels,i.e each column is one voxel
    """

    nrows=fmri.shape[0]
    ncols=fmri.shape[1]
    nslices=fmri.shape[2]
    nframes=fmri.shape[3]

    nvoxels=int(sum(mask.flatten()))

    mask_reshaped=np.reshape(mask,(1,nrows*ncols*nslices))

    ts=np.zeros((nframes,nvoxels))

    for i in range(nframes):
        frame=fmri[:,:,:,i]
        frame=np.reshape(frame,(1,nrows*ncols*nslices))
        ts[i,:]=frame[mask_reshaped==1]

    return ts


def extract_ROIs_from_atlas(mask,in_value):
    """
    extract ROIs from a given atlas
    Input:
    atlas: a 3D matrix with values
    in_value: value to extract
    out_value: value to assign

    Return:
    a 3D matrix with values
    """
    template = nl.image.load_img(mask)
    matrix = template.get_data()
    matrix_reshaped = np.reshape(matrix, (template.shape[0]*template.shape[1]*template.shape[2], 1))
    for voxel in range(template.shape[0]*template.shape[1]*template.shape[2]):
        if matrix_reshaped[voxel]==in_value:
            matrix_reshaped[voxel]= 1
        else:
            matrix_reshaped[voxel]=0


    matrix_reshaped = np.reshape(matrix_reshaped, template.shape)
    template.get_data=matrix_reshaped
    new_image = nib.Nifti1Image(matrix_reshaped, affine=template.affine)


    return new_image


# Listing the ROIs to use down the line
files = glob.glob("/Volumes/WD_Elements/HCP_fmri/ROIs/*atlas2MNI.nii.gz")

# Grabbing the fMRI files to use in the for loop down the line
fmri_files = glob.glob("/Volumes/WD_Elements/BROAD/DTI/new_preprocessed/sub*/dwi/03_bet/*rfMRI_REST1_LR_hp2000_clean.nii.gz")

atlas = '103818_atlas2MNI.nii.gz'
grab_folder = len(files[0]) -  len(atlas)


for z in range(45):
    fmri = nib.load(fmri_files[z])
    fmri = np.array(fmri.dataobj)

    # Looping over the ROIs and extract the timeseries
    matrix=[]
    for x in range(1,235):
        mask = extract_ROIs_from_atlas(files[z], x)
        mask = np.array(mask.dataobj)
        prova = extract_mask_timeseries(mask,fmri)
        media = np.mean(prova,axis=1)
        gigi = stats.zscore(media)
        matrix.append(gigi)


    matrix = np.array(matrix)

    #changing directory
    directory = files[z][0:grab_folder]
    os.chdir(directory)

    df = pd.DataFrame(matrix)
    df.to_csv('Func_atlas_FC_matrix.csv', index=False, header=False)

    # Regressing out the mean signal as an 'easy' way to remove this trend
    tmp = np.mean(matrix,axis=0);

    matrix_2 = np.zeros((matrix.shape[0],matrix.shape[1]));

    for i in range(0,234):
       matrix_2[i,:] = matrix[i,:] - tmp;

    df = pd.DataFrame(matrix_2)
    df.to_csv('Func_atlas_FC_matrix_GSR.csv', index = False, header=False)
