"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import glob

import zipfile
import urllib.request
import shutil
import imageio
import skimage.draw
import cv2
import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class SsndaConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ssnda"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # SSNDa has 4 classes

    WIDTH = 256
    HEIGHT = 160

    SEG_DIRS = ['ball', 'ball0', 'goal_left_left', 'goal_left_right',
                'goal_left_top', 'goal_right_left', 'goal_right_right',
                'goal_right_top', 'NAO#0', 'NAO#1', 'NAO#2', 'NAO#3', 'NAO#4']
    SEG_CLASSES = np.array([2, 2, 1, 1, 1, 1, 1, 1, 4, 3, 3, 4, 4], dtype=np.int32)
    NUMBER_OBSTACLES = len(SEG_DIRS)

############################################################
#  Dataset
############################################################

class SsndaDataset(utils.Dataset):
    def load_ssnda(self, dataset_dir, subset, class_ids=None,
                  class_map=None, auto_download=False):
        """Load a subset of the SSNDa dataset.
        dataset_dir: The root directory of the SSNDa dataset.
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_ssnda: If True, returns the SSNDa object.
        auto_download: Automatically download and unzip SSNDa images and annotations
        """

        # if auto_download is True:
        #     self.auto_download(dataset_dir, subset)

        # ssnda = SSNDa("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        # if subset == "minival" or subset == "valminusminival":
        #     subset = "val"
        # image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # # Load all classes or a subset?
        # if not class_ids:
        #     # All classes
        #     class_ids = sorted(coco.getCatIds())

        # # All images or a subset?
        # if class_ids:
        #     image_ids = []
        #     for id in class_ids:
        #         image_ids.extend(list(coco.getImgIds(catIds=[id])))
        #     # Remove duplicates
        #     image_ids = list(set(image_ids))
        # else:
        #     # All images
        #     image_ids = list(coco.imgs.keys())

        # # Add classes
        # for i in class_ids:
        #     self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add classes.
        self.add_class("ssnda", 1, "goal")
        self.add_class("ssnda", 2, "ball")
        self.add_class("ssnda", 3, "robot_team")
        self.add_class("ssnda", 4, "robot_opponent")

        # Train or validation dataset?
        assert subset in ["train", "val", ""]
        dataset_dir = os.path.join(dataset_dir, "")
        print(dataset_dir)

        # Add images
        images_path = sorted(glob.glob(os.path.join(dataset_dir, "", 'rgb', '*' + '.png')))
        for img_path in images_path:
            self.add_image(
                "ssnda",
                image_id=img_path,
                path=img_path,
                width=SsndaConfig.WIDTH,
                height=SsndaConfig.HEIGHT)
        # if return_ssnda:
        #     return ssnda

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a SSDNa image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "ssnda":
            return super(SsndaDataset, self).load_mask(image_id)

         # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], SsndaConfig.NUMBER_OBSTACLES],
                        dtype=np.uint8)
        rgb_path, img_name = os.path.split(info["path"])
        dataset_dir = os.path.split(rgb_path)[0]

        # Add segmentations to mask
        for i in range(len(SsndaConfig.SEG_DIRS)):
            seg_dir = os.path.join(dataset_dir, 'obstacles_seg', SsndaConfig.SEG_DIRS[i], img_name)
            mask[:, :, i] = self.get_bool_image(seg_dir, info)

        return mask, SsndaConfig.SEG_CLASSES

    def get_bool_image(self, img_dir, info):
        im = None
        try:
            im = imageio.imread(img_dir)[:, :, 0].astype(np.bool)
        except FileNotFoundError:
            im = np.zeros((info["height"], info["width"]), dtype=np.bool)

        return im

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = SsndaDataset()
    dataset_train.load_ssnda(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SsndaDataset()
    dataset_val.load_ssnda(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect SSNDa objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        default='',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SsndaConfig()
    else:
        class InferenceConfig(SsndaConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    load_weights = True
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    elif args.weights == "":
        load_weights = False
    else:
        weights_path = args.weights

    # Load weights
    if load_weights:
        print("Loading weights ", weights_path)
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
