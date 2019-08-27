# -*- coding=UTF-8 -*-

import os
import datetime
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from PIL import Image

from config import config
from utils.compiler import ModelCompiler, Dice
from utils.datasets import ReadCsvImageDataSet
from models.unet import UNet


if __name__ == '__main__':

    opt = config
    print(opt)

    # Experiment index
    exp_cls = "A"
    exp_idx = 1
    exp_name = "%s%03d" % (exp_cls, exp_idx)

    # Validation index
    val_idx = 1 # Or your can use config to modify: val_idx = opt.val_index

    # Configure dataloaders
    transforms_train = [
        transforms.Resize((opt.img_crop_height, opt.img_crop_width), Image.BICUBIC),
        transforms.ToTensor(),
    ]

    transforms_val = [
        transforms.Resize((opt.img_crop_height, opt.img_crop_width), Image.BICUBIC),
        transforms.ToTensor(),
    ]

    train_csv_file = opt.dataset_path + "/" + "list.csv"
    img_container = opt.dataset_path

    # Training dataset read from CSV
    train_dataset = ReadCsvImageDataSet(csv_file_path_=train_csv_file,
                                        transforms_=transforms_train,
                                        image_container_=img_container,
                                        mode='train',
                                        validation_index_=val_idx,
                                        use_grayscale_mask=True)  # True: load mask with grayscale, False: RGB

    # Validation dataset read from CSV
    val_dataset = ReadCsvImageDataSet(csv_file_path_=train_csv_file,
                                      transforms_=transforms_val,
                                      image_container_=img_container,
                                      mode='val',
                                      validation_index_=val_idx,
                                      use_grayscale_mask=True)  # True: load mask with grayscale, False: RGB

    # Get dataset name
    dataset_name = os.path.split(opt.dataset_path)[-1]

    # Get today datetime
    today = datetime.date.today()
    today = "%d%02d%02d" % (today.year, today.month, today.day)

    # Model
    model = UNet(3, 1) # UNet(input_channels, output_channels)
    model_name = model.__class__.__name__

    # Pre-train model
    # load_model_path = r"pretrain_models/UNet_600.pth" # Or your can use config to modify: load_model_path = opt.load_model_path

    # Checkpoint name
    save_ckpt_name = r"%s-%s-%s-(%s)-ep(%d)-bs(%d)-lr(%s)-img_size(%d, %d, %d)-crop_size(%d, %d, %d)-val_index(%d)-dice_loss-hue(%s)" \
                     % (exp_name, today, dataset_name, model_name, opt.n_epochs, opt.batch_size, opt.lr,opt.img_height, opt.img_width,
                        opt.channels, opt.img_crop_height, opt.img_crop_width, opt.channels, val_idx, opt.hue)

    # Training configuration & hyper-parameters
    training_configuration = {
        "validation_index": val_idx,
        "today": today,
        "dataset_name": dataset_name,
        "model": model,
        "model_name": model_name,

        # "load_model_path": load_model_path, # retrain

        "train_dataloader": DataLoader(dataset=train_dataset,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       num_workers=opt.n_cpu),

        "validation_dataloader": DataLoader(dataset=val_dataset,
                                            batch_size=opt.val_batch_size,
                                            shuffle=True,
                                            num_workers=4),  # Thread

        "loss_function": Dice(eps=1.),
        "optimizer": lambda model: torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)),
        "save_ckpt_in_path": save_ckpt_name,
        "tensorboard_path": os.path.join("tf_log", save_ckpt_name),
    }

    train_compiler = ModelCompiler(**training_configuration)
    train_compiler.train()