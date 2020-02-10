import numpy as np
import nilearn as nl
import nibabel as nib
import os
import glob



def extract_ROIs_from_atlas(mask,in_value,out_value):
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
            matrix_reshaped[voxel]= out_value
        else:
            matrix_reshaped[voxel]=0


    matrix_reshaped = np.reshape(matrix_reshaped, template.shape)
    template.get_data=matrix_reshaped
    new_image = nib.Nifti1Image(matrix_reshaped, affine=template.affine)


    return new_image


for z in range(1,235):
    mask = '/Volumes/WD_Elements/HCP_fmri/Yeo.nii'
    directory = '/Volumes/WD_Elements/HCP_fmri/ROIs_2'
    os.chdir(directory)
    ROIs = extract_ROIs_from_atlas(mask,z,1)
    name = os.path.join(directory + '/' + str(z) + '.nii.gz')
    nib.save(ROIs,name)
