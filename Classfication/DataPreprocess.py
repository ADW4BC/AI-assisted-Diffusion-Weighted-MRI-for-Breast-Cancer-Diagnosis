import SimpleITK as sitk
import os
import openpyxl
import csv


def Rigid_registration(fixed_image, moving_image):

    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=400)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-5, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()


    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)


    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))


    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear,
                                     float(0),
                                     moving_image.GetPixelID())

    return moving_resampled





def expand_square_bounding_box(segmentation_image, target_size=64):


    label_statistics = sitk.LabelShapeStatisticsImageFilter()
    label_statistics.Execute(segmentation_image)


    bounding_box = label_statistics.GetBoundingBox(1)

    size = bounding_box[3:5]
    max_size = max(size[:2])

    target_size = max(target_size, max_size)

    extra_size = (target_size - max_size) // 2

    start_point = [bounding_box[0] - extra_size, bounding_box[1] - extra_size, bounding_box[2]]

    expanded_box = start_point + [target_size, target_size, bounding_box[5]]

    return expanded_box



def crop_to_bounding_box(image, bounding_box):

    start_point = bounding_box[:3]
    size = bounding_box[3:6]

    image_size = image.GetSize()


    for i in range(3):
        if start_point[i] < 0:
            return None
        if start_point[i] + size[i] > image_size[i]:
            return None

    cropped_image = sitk.RegionOfInterest(image, size, start_point)

    return cropped_image


def resample_image(image, new_spacing):

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled_image = resampler.Execute(image)
    return resampled_image



def Preprocess():


    segment_dir = r''
    b0_image_dir = r''
    b800_image_dir = r''
    t2_image_dir = r''
    c_image_dir = r''

    # read image
    segment_image = sitk.ReadImage(origin_segment_dir)
    b0_image = sitk.ReadImage(b0_image_dir, sitk.sitkFloat32)
    b800_image = sitk.ReadImage(b800_image_dir, sitk.sitkFloat32)
    t2_image = sitk.ReadImage(t2_image_dir, sitk.sitkFloat32)
    c_image = sitk.ReadImage(c_image_dir, sitk.sitkFloat32)

    # rigid registration
    b0_image_regis = Rigid_registration(c_image, b0_image)
    b800_image_regis = Rigid_registration(c_image, b800_image)
    t2_image_regis = Rigid_registration(c_image, t2_image)

    # resample
    new_spacing = (1.0, 1.0, 1.0)
    resample_segment_image = resample_image(segment_image, new_spacing)
    resample_c_image = resample_image(c_image, new_spacing)
    resample_b0_image = resample_image(b0_image_regis, new_spacing)
    resample_b800_image = resample_image(b800_image_regis, new_spacing)
    resample_t2_image = resample_image(t2_image_regis, new_spacing)


    # crop
    bounding_box = expand_square_bounding_box(resample_segment_image)
    cropped_segmentation = crop_to_bounding_box(resample_segment_image, bounding_box)

    cropped_c = crop_to_bounding_box(resample_c_image, bounding_box)
    cropped_b0 = crop_to_bounding_box(resample_b0_image, bounding_box)
    cropped_b800 = crop_to_bounding_box(resample_b800_image, bounding_box)
    cropped_t2 = crop_to_bounding_box(resample_t2_image, bounding_box)


    # save
    save_path_segmentation = os.path.join(segment_ouput_folder, f"{data}.nii.gz")
    save_path_c = os.path.join(c_output_folder, f"{data}_c.nii.gz")
    save_path_b0 = os.path.join(b0_output_folder, f"{data}_b0.nii.gz")
    save_path_b800 = os.path.join(b800_output_folder, f"{data}_b800.nii.gz")
    save_path_t2 = os.path.join(t2_output_folder, f"{data}_t2.nii.gz")


    # save image
    sitk.WriteImage(cropped_segmentation, save_path_segmentation)
    sitk.WriteImage(cropped_c, save_path_c)
    sitk.WriteImage(cropped_b0, save_path_b0)
    sitk.WriteImage(cropped_b800, save_path_b800)
    sitk.WriteImage(cropped_t2, save_path_t2)



if __name__ == "__main__":
    Preprocess()
