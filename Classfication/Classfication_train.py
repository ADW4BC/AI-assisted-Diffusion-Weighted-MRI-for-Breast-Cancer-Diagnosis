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
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd, RandGaussianNoised, RandFlipd,RandAffined, RandRotated,AddChanneld,ConcatItemsd

import time
import pandas as pd
from sklearn.model_selection import train_test_split
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def GetTrainingData(b0_dir, b800_dir, t2_dir, c_dir, label_dir, seed):
    c_images = sorted(glob.glob(os.path.join(c_dir, "*.nii.gz")))
    b0_images = sorted(glob.glob(os.path.join(b0_dir, "*.nii.gz")))
    b800_images = sorted(glob.glob(os.path.join(b800_dir, "*.nii.gz")))
    t2_images = sorted(glob.glob(os.path.join(t2_dir, "*.nii.gz")))


    df = pd.read_csv(label_dir)
    labels = df['Label'].to_numpy()

    random.seed(seed)
    indices = list(range(len(labels)))
    random.shuffle(indices)

    train_ratio = 0.6
    val_ratio = 0.2

    train_split = int(train_ratio * len(indices))
    val_split = int((train_ratio + val_ratio) * len(indices))

    train_files = [{"b0": b0, "b800": b800, "t2": t2, "c": c, "label": label} for b0, b800, t2, c, label in
                   zip(list(np.array(b0_images)[indices[:train_split]]),
                       list(np.array(b800_images)[indices[:train_split]]),
                       list(np.array(t2_images)[indices[:train_split]]),
                       list(np.array(c_images)[indices[:train_split]]),
                       list(np.array(labels)[indices[:train_split]]))]

    val_files = [{"b0": b0, "b800": b800, "t2": t2, "c": c, "label": label} for b0, b800, t2, c, label in
                 zip(list(np.array(b0_images)[indices[train_split:val_split]]),
                     list(np.array(b800_images)[indices[train_split:val_split]]),
                     list(np.array(t2_images)[indices[train_split:val_split]]),
                     list(np.array(c_images)[indices[train_split:val_split]]),
                     list(np.array(labels)[indices[train_split:val_split]]))]

    test_files = [{"b0": b0, "b800": b800, "t2": t2, "c": c, "label": label} for b0, b800, t2, c, label in
                  zip(list(np.array(b0_images)[indices[val_split:]]),
                      list(np.array(b800_images)[indices[val_split:]]),
                      list(np.array(t2_images)[indices[val_split:]]),
                      list(np.array(c_images)[indices[val_split:]]),
                      list(np.array(labels)[indices[val_split:]]))]

    return train_files, val_files, test_files

def train():

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_files, val_files, test_files = GetTrainingData(r'classification\b0_image',
                                                         r'classification\b800_image',
                                                         r'classification\t2_image',
                                                         r'classification\t1c_image',
                                                         r'classification\label.csv')

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["b0", "b800", "t2", "c"], ensure_channel_first=True),
            ScaleIntensityd(keys=["b0", "b800", "t2", "c"]),
            Resized(keys=["b0", "b800", "t2", "c"], spatial_size=(64, 64, 32)),

            RandGaussianNoised(keys=["b0", "b800", "t2", "c"], prob=0.7),
            ConcatItemsd(keys=["b0", "b800", "t2", "c"], name="img"),

            RandAffined(
                keys=["img"],
                prob=0.9,
                translate_range=(10, 10, 5),  # (-10, 10), (-10, 10), (-5, 5)  in x,y,z
                rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),  # (-10, 10) degrees
                scale_range=(0.1, 0.1, 0.1),  # (1.0 - 0.1, 1.0 + 0.1)
                padding_mode="zeros",
            ),

            RandFlipd(keys=["img"], prob=0.7, spatial_axis=0),
            RandFlipd(keys=["img"], prob=0.7, spatial_axis=1),
            RandFlipd(keys=["img"], prob=0.7, spatial_axis=2),
            # RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),

        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["b0", "b800", "t2", "c"], ensure_channel_first=True),
            ScaleIntensityd(keys=["b0", "b800", "t2", "c"]),
            Resized(keys=["b0", "b800", "t2", "c"], spatial_size=(64, 64, 32)),
            ConcatItemsd(keys=["b0", "b800", "t2", "c"], name="img"),

        ]
    )

    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    # Define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=10, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["label"])

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=43, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=20,  pin_memory=torch.cuda.is_available())


    # start a typical PyTorch training
    val_interval = 5
    best_metric = -1
    best_auc = -1

    best_train_metric = -1
    best_train_auc = -1

    best_metric_epoch = -1
    max_epochs = 400

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=4, out_channels=2, dropout_prob=0.1).to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    auc_metric = ROCAUCMetric()

    writer = SummaryWriter()
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        lr_scheduler.step()
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        writer.add_scalar("train_epoch_loss", epoch_loss, epoch + 1)

        if (epoch + 1) % val_interval == 0:
            model.eval()
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

                # training auc
                y_train_pred = torch.tensor([], dtype=torch.float32, device=device)
                y_train = torch.tensor([], dtype=torch.long, device=device)
                for train_data in train_loader:
                    train_images, train_labels = train_data["img"].to(device), train_data["label"].to(device)
                    y_train_pred = torch.cat([y_train_pred, model(train_images)], dim=0)
                    y_train = torch.cat([y_train, train_labels], dim=0)
                train_acc_value = torch.eq(y_train_pred.argmax(dim=1), y_train)
                train_acc_metric = train_acc_value.sum().item() / len(train_acc_value)
                train_y_onehot = [post_label(i) for i in decollate_batch(y_train, detach=False)]
                train_y_pred_act = [post_pred(i) for i in decollate_batch(y_train_pred)]
                auc_metric(train_y_pred_act, train_y_onehot)
                train_auc_result = auc_metric.aggregate()
                auc_metric.reset()
                del train_y_pred_act, train_y_onehot

                if (acc_metric > best_metric) or (epoch ==max_epochs-1):
                    best_metric = acc_metric
                    best_auc = auc_result
                    best_metric_epoch = epoch + 1

                    best_train_metric = train_acc_metric
                    best_train_auc = train_auc_result

                    current_epoch = epoch + 1

                    torch.save(model.state_dict(), os.path.join('dwi_t2_c_model',str(current_epoch)+"_best_metric_model_classification3d_dict.pth"))
                    print("saved new best metric model")


                print(
                    "current epoch:{} current val accuracy: {:.4f} current val AUC: {:.4f} best val accuracy: {:.4f} best val auc: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_metric, best_auc, best_metric_epoch
                    )
                )

                print(
                    "current epoch: {} current train accuracy: {:.4f} current train AUC: {:.4f}  best train accuracy: {:.4f} best train auc: {:.4f} at epoch {}".format(
                        epoch + 1, train_acc_metric, train_auc_result, best_train_metric, best_train_auc, best_metric_epoch
                    )
                )

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    train()