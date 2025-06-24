# AI-assisted Diffusion Weighted MRI for Breast Cancer Diagnosis: A Multicenter, Multidimensional Validation Study

[![License: CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)  
![Python Versions](https://img.shields.io/badge/python-3.10-blue
)

## License

This code is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## Background

This is the repository for our work hosting some materials that helps you prepare your dataset, train the model and inference.
## Get Started

### Install MainRequirements

```sh
cd nnunet
pip install -e .
```
### Install dynamic-network 

```sh
cd dynamic-network-architectures
pip install -e .
```



### Data preparation

After Install the environment, we first need to create folders `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`.
And add the paths of the three folders to the environment variables.
```sh
export nnUNet_raw="/**/nnUNet_raw"
export nnUNet_preprocessed="/**/nnUNet_preprocessed"
export nnUNet_results="/**/nnUNet_results"
```
The raw dataset is placed in the nnUNet_raw folder 
```
nnUNet_raw/
├── Dataset001_JM
├── Dataset002_SJZ
├── ....

```
NIfTI-formatted image and json files were used for model training and prediction. Please organize the image and mask files in the following structure.

```
nnUNet_raw/Dataset001_JM/
├── dataset.json
├── imagesTr
    │── Breast_001_0000.nii.gz
    │── Breast_002_0000.nii.gz
    │── Breast_003_0000.nii.gz
├── imagesTs
    │── Breast_998_0000.nii.gz
    │── Breast_999_0000.nii.gz
    │── Breast_1000_0000.nii.gz
├── labelsTr
    │── Breast_001.nii.gz
    │── Breast_001.nii.gz
    │── .....
```
dataset.json like this 

```json
{
    "description": "",
    "labels": {
        "background": 0,
        "tumor": 1
    },
    "licence": "hands off!",
    "name": "Dataset002_PkuC",
    "numTraining": 1286,
    "reference": "",
    "release": "0.0",
    "tensorImageSize": "4D",
    "file_ending": ".nii.gz",
    "channel_names": {
        "0": "M"
    }
}
```

## Model Training and Validation
1. Modify the training configuration in `\nnunetv2\training\nnUNetTrainernnUNetTrainer.py`.
2. The preprocessed files of the original dataset are saved in `nnUNet_preprocessed`.
```sh
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```
3. Open `nnUNetPlans.json` in the `nnUNet_preprocessed` folder and find the 'architecture' in the 'configurations' to modify the `network_class_name`
```sh
dynamic_network_architectures.architectures.shuffleattunet.ShuffleAttPlainConvUNet
```
4. run train command and save results in `nnUNet_results`.
```sh
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```
5. run predict command to perform inference,and each image inference was performed 5 times
```sh
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```



## Classification model 

### Install MainRequirements

```
SimpleITK
openpyxl
pandas
torch
monai
sklearn
```

## data preprocessing

run `Preprocess` in `DataPreprocess.py` module to perform data preprocessing operations. Preprocessing includes five steps: rigid registration, resampling, cropping, normalization, and concatenation. 

You need to set the path to the image for different modalities in the Preprocess function.

```python
segment_dir = r''
b0_image_dir = r''
b800_image_dir = r''
t2_image_dir = r''
c_image_dir = r''
```



## Model Training and Validation

1. Before model training and validation, please organize the preprocessed images and mask files according to the following structure. All images are NIfTI format, label.csv stores the label corresponding to each data.

```
/data_set/
├── classification
    │── t1c_image
    │    ├── patient_1_c.nii.gz
    │    ├── patient_2_c.nii.gz
    │    └── ......
    │── b0_image
    │    ├── patient_1_b0.nii.gz
    │    ├── patient_2_b0.nii.gz
    │    └── ......
    │── b800_image
    │    ├── patient_1_b800.nii.gz
    │    ├── patient_2_b800.nii.gz
    │    └── ......
    │── t2_image
    │    ├── patient_1_t2.nii.gz
    │    ├── patient_2_t2.nii.gz
    │    └── ......
    └── label.csv
  
```

2. run `GetTrainingData` in `Classfication_train` module to read data and randomly divide it into training set, test set and validation set.

3. run `train` in `Classfication_train` module to perform data augmentation and start training the model. 

4. run `external_validation` in `Classfication_external_validate` module to external validate the model in holdout datasets.

5. run `Grad_CAM` in `Classfication_external_validate` module to generate Gradient-Weighted Class Activation Map (Grad-CAM) heatmaps.
