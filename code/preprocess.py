import os
import time

import SimpleITK as sitk
import pandas as pd


def load_dataset(data_file_path):
    df = pd.read_csv(data_file_path)

    return df[['filename', 'finding']]


def apply_n4itk(image):
    a = time.time()
    mask_image = sitk.OtsuThreshold(image, 0, 1, 200)
    input_image = sitk.Cast(image, sitk.sitkFloat32)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    # number_fitting_levels=4

    # corrector.SetMaximumNumberOfIterations(number_fitting_levels)

    output = corrector.Execute(input_image, mask_image)
    casted_output = sitk.Cast(output, sitk.sitkUInt8)
    b = time.time()
    print('time taken is %.2f seconds' % (b-a))
    return casted_output


def read_image(image_path):
    return sitk.ReadImage(image_path, sitk.sitkInt8)


def separate_dataset(df, images_dir_path):
    for index in range(df.shape[0]):
        type = str(df.iloc[index]['finding'])
        print(index, 'loading...')
        if not os.path.exists(type):
            os.mkdir('./{}'.format(type))
        filename = df.iloc[index]['filename']
        image_path = os.path.join(images_dir_path, filename)
        bias_corrected_image = apply_n4itk(read_image(image_path))
        sitk.WriteImage(bias_corrected_image, './{}/{}'.format(type, filename))


if __name__ == '__main__':
    current_path = os.path.join(os.path.dirname(os.path.abspath(os.path.curdir)), 'dataset', 'covid-chestxray-dataset')
    print(current_path)
    separate_dataset(load_dataset(os.path.join(current_path, 'metadata.csv')), os.path.join(current_path, 'images'))
