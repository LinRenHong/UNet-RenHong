
import random
import os
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from PIL import Image

from config import config

opt = config


class ReadCsvImageDataSet(Dataset):
    def __init__(self,csv_file_path_, image_container_, transforms_, image_format_=None, mode='train', validation_index_=None, use_grayscale_mask=False):

        self.csv_flie_path = csv_file_path_
        self.image_container = image_container_
        self.image_format = image_format_
        self.transforms = transforms.Compose(transforms_)
        self.condition = mode
        self.validation_index = validation_index_
        self.use_grayscale_mask = use_grayscale_mask

        self.image_paths = None
        self.vf_patch_paths = None
        self.labels = None
        self.fold_index = None
        self.label_names = None
        self.select_indices = None


        self.df = self._read_csv()

        self.image_A_paths = self.df["Image"]
        self.image_B_paths = self.df["Mask"]
        self.labels = np.array(self.df["Label"])
        self.fold_index = np.array(self.df["FoldIndex"])


        self._check_img_A()
        self._check_img_B()

        self._set_select_indices_by_validation_index()
        self._set_data_by_validation_index()

    def __getitem__(self, item):
        '''
        Retrieve item in dataset, including reading images & transforming images
        :param item: index for retrieving items
        :return:
            data_dict: <dict> the dictionary containing data (Keys: Image, Label)
        '''

        img_A = Image.open(self.image_A_paths[item]).convert('RGB')
        img_B = Image.open(self.image_B_paths[item]).convert('RGB')

        if self.use_grayscale_mask:
            ### let mask become 0 and 255 ###
            img_B = np.array(img_B)
            img_B[img_B <= 100] = 0
            img_B[img_B > 100] = 255
            img_B = Image.fromarray(img_B)
            ### let mask become 0 and 255 ###


        ### Random horizontal flip ###
        if random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        ### Random horizontal flip ###

        ### train ###
        if self.condition.strip() in ["train", "Train", "TRAIN"]:
            ### Resize ###
            resize = transforms.Resize(size=(opt.img_height, opt.img_width))
            img_A = resize(img_A)
            img_B = resize(img_B)
            ### Resize ###

            ### Random crop ###
            i, j, h, w = transforms.RandomCrop.get_params(img_A, output_size=(opt.img_crop_height, opt.img_crop_width))
            img_A = transforms.functional.crop(img_A, i, j, h, w)
            img_B = transforms.functional.crop(img_B, i, j, h, w)
            ### Random crop ###

            if random.random() < 0.5:
                ### Random rotate ###
                random_angle = random.randint(-90, 90)
                img_A = img_A.rotate(random_angle)
                img_B = img_B.rotate(random_angle)
                ### Random rotate ###

            img_A = transforms.ColorJitter(
                                            # brightness=(32./255.),
                                            # contrast=0.5,
                                            # saturation=0.5,
                                            hue=opt.hue)(img_A)

        ### data augmentation ###

        img_A = self.transforms(img_A)
        img_B = self.transforms(img_B)

        if self.use_grayscale_mask:
            ### mask convert to gray scale ###
            img_B = transforms.ToPILImage()(img_B)
            img_B = transforms.Grayscale(1)(img_B)
            img_B = transforms.ToTensor()(img_B)

        return {"A": img_A, "B": img_B}


    def __len__(self):
        return len(self.image_A_paths)


    def _check_img_A(self):
        '''
        Check whether images exist in folder or not
        Note: all images should be in image container
        '''

        # check whether image format is already embedded in image paths (check ".")
        if self.image_format is None and "." not in self.image_A_paths[0]:

            raise FileNotFoundError("There no image format specified in image path {}. "
                                    "Please specifiy image format".format(self.image_A_paths[0]))
        elif self.image_format is not None and "." not in self.image_A_paths[0]:

            print("Concatenate image format into image paths ...")

            self.image_A_paths = np.array([os.path.join(self.image_container,
                                                      image_path+"."+self.image_format)
                                         for image_path in self.image_A_paths])
        else:
            self.image_A_paths = np.array([os.path.join(self.image_container,
                                                      image_path) for image_path in self.image_A_paths])

        for path in self.image_A_paths:

            if not os.path.isfile(path):

                raise FileNotFoundError("Image file - {} not found! Please check!".format(path))

        print("All images exist in folder A({}) - {}".format(self.condition, self.image_container))


    def _check_img_B(self):
        '''
        Check whether images exist in folder or not
        Note: all images should be in image container
        '''

        # check whether image format is already embedded in image paths (check ".")
        if self.image_format is None and "." not in self.image_B_paths[0]:

            raise FileNotFoundError("There no image format specified in image path {}. "
                                    "Please specifiy image format".format(self.image_B_paths[0]))
        elif self.image_format is not None and "." not in self.image_B_paths[0]:

            print("Concatenate image format into image paths ...")

            self.image_B_paths = np.array([os.path.join(self.image_container,
                                                      image_path+"."+self.image_format)
                                         for image_path in self.image_B_paths])
        else:
            self.image_B_paths = np.array([os.path.join(self.image_container,
                                                      image_path) for image_path in self.image_B_paths])

        for path in self.image_B_paths:

            if not os.path.isfile(path):

                raise FileNotFoundError("Image file - {} not found! Please check!".format(path))

        print("All images exist in folder B({}) - {}".format(self.condition, self.image_container))



    def _set_select_indices_by_validation_index(self):
        '''
        Select images by validation index and fold index depending on condition (Train / Validation / Test)
        :return:
        '''

        if self.validation_index is not None and self.fold_index is not None:

            if self.condition.strip() in ["train", "Train", "TRAIN"]:

                indices = self.fold_index != self.validation_index

            else:

                indices = self.fold_index == self.validation_index

            self.select_indices = indices

    def _set_data_by_validation_index(self):
        '''
        Select images by validation index and fold index depending on condition (Train / Validation / Test)
        :return:
        '''

        if self.select_indices is not None:

            self.image_A_paths = self.image_A_paths[self.select_indices]
            self.image_B_paths = self.image_B_paths[self.select_indices]
            self.labels = self.labels[self.select_indices] if self.labels is not None else None
            self.label_names = self.label_names[self.select_indices] if self.label_names is not None else None

    def _read_csv(self):

        try:
            df = pd.read_csv(self.csv_flie_path)
            return df

        except FileNotFoundError as e:
            print(e, "\nPlease check the csv filepath!")



if __name__ == "__main__":
    dataset_name = os.path.split(opt.dataset_path)[-1]
    train_csv_file = opt.dataset_path + "/" + "list.csv"
    img_container = opt.dataset_path

    # Configure dataloaders
    transforms_ = [
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
    ]


    for i in range(1, 6):
        train_dataset = ReadCsvImageDataSet(csv_file_path_=train_csv_file,
                                            transforms_=transforms_,
                                            image_container_=img_container,
                                            mode='val',
                                            validation_index_=i)

        # train_dataset = ImageDataset("../../data/facades", transforms_=transforms_)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=8)


        data_dict = next(iter(train_loader))
        # print(data_dict)
        print(len(train_dataset))


