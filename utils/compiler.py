
import os
import time
import datetime
import sys
import random
import numpy as np
import collections

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from tensorboardX import SummaryWriter
from math import log10
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from config import config
from models.unet import UNet
from models.unet import weights_init_normal

opt = config


class ModelCompiler(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.today = kwargs.get("today", None)
        self.dataset_name = kwargs.get("dataset_name", None)
        self.model_name = kwargs.get("model_name", None)
        self.val_idx = kwargs.get("validation_index", None)
        self.save_ckpt_name = kwargs.get("save_ckpt_in_path", None)
        self.tb_log_path = kwargs.get("tensorboard_path", None)

        # Training dataloader
        self.train_dataloader = kwargs.get("train_dataloader", None)

        # Validation dataloader
        self.val_dataloader = kwargs.get("validation_dataloader", None)

        # Model
        self.model = kwargs.get("model", None)

        # Pre-train model
        self.load_model_path = kwargs.get("load_model_path", None)

        # Loss function
        self.criterion = kwargs.get("loss_function", None)

        # Optimizer
        self.optimizer = kwargs.get("optimizer", None)
        if self.optimizer is not None:
            self.optimizer = self.optimizer(self.model)

        # Use GPU
        self.isCuda = True if torch.cuda.is_available() else False
        # Tensor type
        self.tensor_type = torch.cuda.FloatTensor if self.isCuda else torch.FloatTensor

        self.results_dir = "results"
        self.save_images_dir = os.path.join(self.results_dir, "images")
        self.save_models_dir = os.path.join(self.results_dir, "saved_models")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.save_images_dir, exist_ok=True)
        os.makedirs(self.save_models_dir, exist_ok=True)


    def train(self):

        os.makedirs(os.path.join(self.save_images_dir, "%s" % self.save_ckpt_name), exist_ok=True)
        os.makedirs(os.path.join(self.save_models_dir, "%s" % self.save_ckpt_name), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.results_dir, self.tb_log_path))

        # If use GPU
        if self.isCuda:
            self.model = self.model.cuda()
            self.criterion.cuda()

        # If want to retrain
        if opt.epoch != 0:
            # Load pretrained models
            print("Loading model from: [%s]" % self.load_model_path)
            self.model.load_state_dict(torch.load("%s" % self.load_model_path))
            print("Start retrain...")
        else:
            # Initialize weights
            self.model.apply(weights_init_normal)
            print("Start training...")


        prev_time = time.time()
        self._best_train_dice = 0.0

        for epoch in range(opt.epoch, opt.n_epochs):

            total_train_dice = 0.0
            total_train_psnr = 0.0

            for i, batch in enumerate(self.train_dataloader):
                # Model inputs
                real_A = Variable(batch["A"].type(self.tensor_type))
                real_B = Variable(batch["B"].type(self.tensor_type))
                masks_pred = self.model(real_A)

                # Calculate loss
                self.loss = self.criterion(masks_pred, real_B)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                # Dice
                train_dice = ModelEvaluation.calculate_dice_coefficient(masks_pred, real_B, eps=1.)
                total_train_dice = total_train_dice + train_dice

                # PSNR
                train_psnr = calculate_PSNR(masks_pred, real_B)
                total_train_psnr = total_train_psnr + train_psnr

                # Determine approximate time left
                batches_done = epoch * len(self.train_dataloader) + i
                batches_left = opt.n_epochs * len(self.train_dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f, dice: %f, PSNR: %.4f dB] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(self.train_dataloader),
                        self.loss.item(),
                        train_dice.item(),
                        train_psnr,
                        time_left,
                    )
                )

            # calculate average Dice and PSNR
            self._avg_train_dice = total_train_dice / len(self.train_dataloader)
            self._avg_train_psnr = total_train_psnr / len(self.train_dataloader)

            # Save model
            self.save_model(epoch=epoch)

            # write to TensorBoard
            self.write_to_tensorboard(epoch=epoch, condition="train")

            # Validate
            self.validate(epoch_done=epoch, is_save_image=True)


    def validate(self, epoch_done, is_save_image=True):
        total_val_dice = 0.0
        total_val_psnr = 0.0

        temp_real_A = 0
        temp_real_B = 0
        temp_fake_B = 0

        # Random sample 10 images (except first image)
        sample_num = 9
        want_sample = random.sample(range(1, len(self.val_dataloader)), sample_num)

        # Iterate validation set
        for i, batch in enumerate(self.val_dataloader):

            real_A = Variable(batch["A"].type(self.tensor_type))
            real_B = Variable(batch["B"].type(self.tensor_type))
            fake_B = self.model(real_A)

            val_dice = ModelEvaluation.calculate_dice_coefficient(fake_B, real_B, eps=1.)
            total_val_dice = total_val_dice + val_dice

            val_psnr = calculate_PSNR(fake_B, real_B)
            total_val_psnr = total_val_psnr + val_psnr


            if is_save_image and (i == 0 or i in want_sample):

                real_A_cpu = real_A.cpu().clone()
                real_B_cpu = real_B.cpu().clone()
                fake_B_cpu = fake_B.cpu().clone()

                # Mask is grayscale
                if real_B_cpu.size(1) == 1:
                    ### let GrayScale dim 1 to 3 ###
                    image_mask = transforms.ToPILImage()(real_B_cpu[0])
                    predict_mask = transforms.ToPILImage()(fake_B_cpu[0])

                    image_mask = transforms.Grayscale(3)(image_mask)
                    predict_mask = transforms.Grayscale(3)(predict_mask)

                    image_mask = transforms.ToTensor()(image_mask)
                    predict_mask = transforms.ToTensor()(predict_mask)
                    ### let GrayScale dim 1 to 3 ###

                # Mask is RGB
                else:
                    image_mask = real_B_cpu[0]
                    predict_mask = fake_B_cpu[0]

                # If is first element
                if i == 0:
                    temp_real_A = real_A_cpu
                    temp_real_B = image_mask.unsqueeze_(0)
                    temp_fake_B = predict_mask.unsqueeze_(0)
                else:
                    temp_real_A = torch.cat((temp_real_A, real_A_cpu), dim=0)
                    temp_real_B = torch.cat((temp_real_B, image_mask.unsqueeze_(0)), dim=0)
                    temp_fake_B = torch.cat((temp_fake_B, predict_mask.unsqueeze_(0)), dim=0)

        # calculate average Dice and PSNR
        self._avg_val_dice = total_val_dice / len(self.val_dataloader)
        self._avg_val_psnr = total_val_psnr / len(self.val_dataloader)

        # Write to TensorBoard
        self.write_to_tensorboard(epoch=epoch_done, condition="val")


        if is_save_image:
            img_sample = torch.cat((temp_real_A.data, temp_fake_B.data, temp_real_B.data), -2)
            save_image(img_sample, os.path.join(self.save_images_dir, "%s/ep(%s).png" % (self.save_ckpt_name, epoch_done)), nrow=5, normalize=True)


    def test(self, is_save_image=True):

        assert self.load_model_path is not None, "Pre-train model is not found! Please check the pre-train model path"
        print("Loading model from: [%s]" % self.load_model_path)
        self.model.load_state_dict(torch.load("%s" % self.load_model_path))

        # If use GPU
        if self.isCuda:
            self.model = self.model.cuda()

        total_val_dice = 0.0
        total_val_psnr = 0.0

        temp_real_A = 0
        temp_real_B = 0
        temp_fake_B = 0

        # Random sample 10 images (except first image)
        sample_num = 9
        want_sample = random.sample(range(1, len(self.val_dataloader)), sample_num)

        print("Start inference...")
        # Iterate validation set
        for i, batch in enumerate(self.val_dataloader):

            real_A = Variable(batch["A"].type(self.tensor_type))
            real_B = Variable(batch["B"].type(self.tensor_type))
            fake_B = self.model(real_A)

            val_dice = ModelEvaluation.calculate_dice_coefficient(fake_B, real_B, eps=1.)
            total_val_dice = total_val_dice + val_dice

            val_psnr = calculate_PSNR(fake_B, real_B)
            total_val_psnr = total_val_psnr + val_psnr

            if is_save_image and (i == 0 or i in want_sample):

                real_A_cpu = real_A.cpu().clone()
                real_B_cpu = real_B.cpu().clone()
                fake_B_cpu = fake_B.cpu().clone()

                # Mask is grayscale
                if real_B_cpu.size(1) == 1:
                    ### let GrayScale dim 1 to 3 ###
                    image_mask = transforms.ToPILImage()(real_B_cpu[0])
                    predict_mask = transforms.ToPILImage()(fake_B_cpu[0])

                    image_mask = transforms.Grayscale(3)(image_mask)
                    predict_mask = transforms.Grayscale(3)(predict_mask)

                    image_mask = transforms.ToTensor()(image_mask)
                    predict_mask = transforms.ToTensor()(predict_mask)
                    ### let GrayScale dim 1 to 3 ###

                # Mask is RGB
                else:
                    image_mask = real_B_cpu[0]
                    predict_mask = fake_B_cpu[0]

                # If is first element
                if i == 0:
                    temp_real_A = real_A_cpu
                    temp_real_B = image_mask.unsqueeze_(0)
                    temp_fake_B = predict_mask.unsqueeze_(0)
                else:
                    temp_real_A = torch.cat((temp_real_A, real_A_cpu), dim=0)
                    temp_real_B = torch.cat((temp_real_B, image_mask.unsqueeze_(0)), dim=0)
                    temp_fake_B = torch.cat((temp_fake_B, predict_mask.unsqueeze_(0)), dim=0)

        # calculate average Dice and PSNR
        self._avg_val_dice = total_val_dice / len(self.val_dataloader)
        self._avg_val_psnr = total_val_psnr / len(self.val_dataloader)

        print("Dice: {}".format(self._avg_val_dice))
        print("PSNR: {}".format(self._avg_val_psnr))

        if is_save_image:
            img_sample = torch.cat((temp_real_A.data, temp_fake_B.data, temp_real_B.data), -2)
            save_image(img_sample, os.path.join(self.save_images_dir, "%s-test(%s)-%s-val_index(%d).png" % (self.today, self.dataset_name, self.model_name, self.val_idx)), nrow=5, normalize=True)



    def save_model(self, epoch):
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            print("\nSave model to [%s] at %d epoch\n" % (self.save_ckpt_name, epoch))
            torch.save(self.model.state_dict(), os.path.join(self.save_models_dir, "%s/%s_%d.pth" % (self.save_ckpt_name, self.model_name, epoch)))

        # Save best model
        if self._avg_train_dice > self._best_train_dice:
            self._best_train_dice = self._avg_train_dice
            print("\nSave best model to [%s]\n" % self.save_ckpt_name)
            torch.save(self.model.state_dict(), os.path.join(self.save_models_dir, "%s/best_%s_%d.pth" % (self.save_ckpt_name, self.model_name, epoch)))

        # Save latest model
        if epoch == (opt.n_epochs - 1):
            print("\nSave latest model to [%s]\n" % self.save_ckpt_name)
            torch.save(self.model.state_dict(), os.path.join(self.save_models_dir, "%s/%s_%d.pth" % (self.save_ckpt_name, self.model_name, opt.n_epochs)))



    def write_to_tensorboard(self, epoch, condition):
        if self.writer is not None:

            if condition.strip() in ["train", "Train", "TRAIN"]:
                self.writer.add_scalar(tag='Loss', scalar_value=self.loss.item(), global_step=epoch)
                self.writer.add_scalar(tag='Dice_train', scalar_value=self._avg_train_dice.item(), global_step=epoch)
                self.writer.add_scalar(tag='PSNR_train', scalar_value=self._avg_train_psnr, global_step=epoch)

            elif condition.strip() in ["val", "Val", "VAL"]:
                self.writer.add_scalar(tag='Dice_val', scalar_value=self._avg_val_dice.item(), global_step=epoch)
                self.writer.add_scalar(tag='PSNR_val', scalar_value=self._avg_val_psnr, global_step=epoch)

            else:
                print("Please specify condition: [\"train\", \"Train\", \"TRAIN\"] or [\"val\", \"Val\", \"VAL\"]")
        else:
            print("Writer is None!")


class ModelEvaluation(object):

    @classmethod
    def calculate_dice_coefficient(cls, predictions, labels, eps=1e-5, input_threshold=0.5, label_threshold=0.5,
                                   reduce=True, verbose=0):
        '''
        Calculate dice coefficient
        :param predictions: <tensor> shape-(number of examples, vector_size or image_size)
            Note: it is the output from model's softmax, tanh or sigmoid layer before binary thresholding
        :param labels: <tensor> shape-(number of examples, vector_size or image_size)
            Each vector or image from each example in labels contains 0s & 1s, which indicate the target area of interest
        :param eps: <float>, threshold for binary thresholding
        :return:
            dice coefficient
        '''

        dice = Dice(eps=eps)

        dice_coefficient = dice(predictions, labels,
                                input_threshold=input_threshold, label_threshold=label_threshold,
                                return_score=True, reduce=reduce)

        if verbose:
            print("Dice coefficient:", dice_coefficient)

        return dice_coefficient



### Dice ###

class Dice(_Loss):

    def __init__(self, weight=None, eps=1e-5, type="sorensen"):
        '''
        :param weight: overall weight for dice loss
        :param eps: epsilon value for smoothing
        :param type: default to "sorensen"
        '''
        self.weight = 1 if weight is None else weight
        self.eps = eps
        self.type = type

        super(Dice, self).__init__()


    def forward(self, input, labels, input_threshold=0.5, label_threshold=0.5, return_score=False, reduce=True):
        '''
        Calculate Dice Coefficient or Dice Coefficient Loss depending on the parameter return_score
        :param input: <tensor> shape-(number of examples, vector_size or image_size)
            Note: it is the output from model's softmax, tanh or sigmoid layer before binary thresholding
        :param labels: <tensor> shape-(number of examples, vector_size or image_size)
            Each vector or image from each example in labels contains 0s & 1s, which indicate the target area of interest
        :param threshold: <float>, threshold for binary thresholding
        :param return_score: return Dice Coefficient or Dice Coefficient loss (Default to False)
        :return:
            loss: Dice Coefficient loss loss if return_score == False
            score: Dice Coefficient if return_score == True
        '''

        input = input.cpu()
        labels = labels.cpu()

        # check size of the input
        if input.size(2) != labels.size(2):

            input = torch.interpolate(input, labels.size(2))

        # reshape input & labels
        number_of_examples = input.size(0)
        input = input.view(number_of_examples, -1)
        labels = labels.view(number_of_examples, -1)

        # thresold the input
        thresholded_input = torch.where(input >= input_threshold,
                                        torch.ones(input.size()),
                                        torch.zeros(input.size()))

        thresholded_labels = torch.where(labels >= label_threshold,
                                         torch.ones(input.size()),
                                         torch.zeros(input.size()))

        input_op = thresholded_input if return_score else input

        overlap = thresholded_labels.mul(input_op).sum(dim=1)

        if self.type == "sorensen":

            input_sum = input_op.sum(dim=1)
            labels_sum = thresholded_labels.sum(dim=1)

        dice = ((2. * overlap + self.eps) / (input_sum + labels_sum + self.eps))

        if reduce:

            dice = dice.mean()

        if return_score:

            return dice

        else:

            return self.weight*(1.-dice)



### Elastic Transform ###

class ElasticTransform(object):
    """Apply elastic transformation on a numpy.ndarray (H x W x C)
    """

    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image):
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma)
        else:
            sigma = self.sigma
        return elastic_transform(image, alpha=alpha, sigma=sigma)


def elastic_transform(image, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    return result


def random_num_generator(config, random_state=np.random):
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    else:
        print(config)
        raise Exception('unsupported format')
    return ret



### PSNR ###

def calculate_PSNR(fake_img, real_img):
    criterionMSE = torch.nn.MSELoss()
    mse = criterionMSE(fake_img, real_img)
    psnr = 10 * log10(1 / mse.item())
    return psnr


if __name__ == '__main__':

    opt = config
    model = UNet(3, 1)
    # print(model)
    x = torch.randn(1, 3, opt.img_crop_height, opt.img_crop_width)
    result = model(x)
    print("Output Image: {}".format(result))
