""" config.py
"""
import argparse

parser = argparse.ArgumentParser("UNet")
parser.add_argument("--load_model_path", type=str, default=r"YOUR_MODEL_PATH", help="model path")

parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
parser.add_argument("--dataset_path", type=str, default=r"dataset/csv_and_image", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--val_batch_size", type=int, default=1, help="size of the validation batches")
parser.add_argument("--val_index", type=int, default=1, help="index of the validation dataset")

parser.add_argument("--lr", type=float, default=1e-5, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

'''--------------------------------------------- hue --------------------------------------------'''
parser.add_argument("--hue", type=float, default=0.15, help="value of hue")
'''--------------------------------------------- hue --------------------------------------------'''

# image size
parser.add_argument("--img_height", type=int, default=286, help="size of image height")
parser.add_argument("--img_width", type=int, default=286, help="size of image width")
parser.add_argument("--img_crop_height", type=int, default=256, help="size of image height after crop")
parser.add_argument("--img_crop_width", type=int, default=256, help="size of image width after crop")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")

### parse and save config ###
config = parser.parse_args()
# config, _ = parser.parse_known_args()
