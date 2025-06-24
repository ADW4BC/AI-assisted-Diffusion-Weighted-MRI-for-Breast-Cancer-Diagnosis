import logging
import os
import sys
import glob
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd, RandGaussianNoised, RandFlipd,RandAdjustContrastd, RandRotated,AddChanneld,ConcatItemsd
import torch.nn.functional as F
import time
import pandas as pd
from torch.autograd import Variable


def GetValidation_data(data_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # the path of dataset
    c_data_dir = os.path.join(data_dir,'c')
    b0_data_dir = os.path.join(data_dir, 'b0')
    b800_data_dir = os.path.join(data_dir, 'b800')
    t2_data_dir = os.path.join(data_dir, 't2')
    label_dir = os.path.join(data_dir, 'label.csv')


    c_images = sorted(glob.glob(os.path.join(c_data_dir, "*.nii.gz")))
    b0_images = sorted(glob.glob(os.path.join(b0_data_dir, "*.nii.gz")))
    b800_images = sorted(glob.glob(os.path.join(b800_data_dir, "*.nii.gz")))
    t2_images = sorted(glob.glob(os.path.join(t2_data_dir, "*.nii.gz")))

    # 2 binary labels for breast classification
    df = pd.read_csv(label_dir)
    labels = df['Label'].to_numpy()

    val_files = [{"b0": b0, "b800": b800, "t2": t2, "c": c, "label": label} for b0, b800, t2, c, label in
                 zip(b0_images[:], b800_images[:], t2_images[:], c_images[:], labels[:])]

    val_transforms = Compose(
        [
            LoadImaged(keys=["b0", "b800", "t2", "c"], ensure_channel_first=True),
            ScaleIntensityd(keys=["b0", "b800", "t2", "c"]),
            Resized(keys=["b0", "b800", "t2", "c"], spatial_size=(64, 64, 32)),
            ConcatItemsd(keys=["b0", "b800", "t2", "c"], name="img"),

        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=100, num_workers=4, pin_memory=torch.cuda.is_available())

    return val_loader

def external_validation(data_dir):

    GetValidation_data(data_dir)

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=2, out_channels=2, dropout_prob=0.1).to(device)


    checkpoint = torch.load('best_metric_model_classification3d_dict.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    auc_metric = ROCAUCMetric()


    with torch.no_grad():

        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
            y_pred = torch.cat([y_pred, model(val_images)], dim=0)
            y = torch.cat([y, val_labels], dim=0)

        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
        auc_metric(y_pred_act, y_onehot)
        auc_result = auc_metric.aggregate()
        auc_metric.reset()

        del y_pred_act, y_onehot
        print("auc: ", auc_result)

def Grad_CAM(data_dir):

    GetValidation_data(data_dir)

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=4, out_channels=2, dropout_prob=0.1).to(device)


    checkpoint = torch.load(r'  ')
    model.load_state_dict(checkpoint)
    model.eval()

    model_name = []
    for name, _ in model.named_modules():
        model_name.append(name)

    for name in reversed (model_name[100:300]):
        # if "features.denseblock2.denselayer4.layers.relu2" in name:
        if "relu2" in name:
            print(name)
            cam = GradCAM(nn_module=model, target_layers=name)
            index = 0
            for val_data in val_loader:
                val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                result = cam(x=val_images,class_idx=1)
                cam_image = sitk.GetImageFromArray(1- result.squeeze().cpu().numpy())

                path = os.path.join(r'\Grad-CAM\dwi+t2+c', name)
                if not os.path.exists(path):
                    os.makedirs(path)

                image_id = val_files[index]['b0'].split("\\")[-1].split("_b0")[0]
                sitk.WriteImage(cam_image, os.path.join(path, image_id +"_cam_image.nii.gz"))



if __name__ == "__main__":
    external_validation()