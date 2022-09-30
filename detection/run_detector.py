r"""
Module to run an animal detection model on images.

The main function in this script also renders the predicted
bounding boxes on images and saves the resulting images (with bounding boxes).

This script is not a good way to process lots of images (tens of thousands,
say). It does not facilitate checkpointing the results so if it crashes you
would have to start from scratch. If you want to run a detector (e.g., ours)
on lots of images, you should check out:

1) run_detector_batch.py (for local execution)

2) https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing
   (for running large jobs on Azure ML)

To run this script, we recommend you set up a conda virtual environment
following instructions in the Installation section on the main README, using
`environment-detector.yml` as the environment file where asked.

This is a good way to test our detector on a handful of images and get
super-satisfying, graphical results.  It's also a good way to see how fast a
detector model will run on a particular machine.

If you would like to *not* use the GPU on the machine, set the environment
variable CUDA_VISIBLE_DEVICES to "-1".

If no output directory is specified, writes detections for c:\foo\bar.jpg to
c:\foo\bar_detections.jpg.

This script will only consider detections with > 0.005 confidence at all times.
The `threshold` you provide is only for rendering the results. If you need to
see lower-confidence detections, you can change
DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD.

Reference:
https://github.com/tensorflow/models/blob/master/research/object_detection/inference/detection_inference.py
"""

#%% Constants, imports, environment

import argparse
from copy import deepcopy
import glob
from operator import truediv
import os
import statistics
import sys
import time
import warnings
import pandas as pd
import json

import humanfriendly
from tqdm import tqdm

from pathlib import Path

import visualization.visualization_utils as viz_utils

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning)

# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings('ignore', 'Metadata warning', UserWarning)

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

# Useful hack to force CPU inference
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# An enumeration of failure reasons
FAILURE_INFER = 'Failure inference'
FAILURE_IMAGE_OPEN = 'Failure image access'

# Number of decimal places to round to for confidence and bbox coordinates
CONF_DIGITS = 3
COORD_DIGITS = 4

# Label mapping for MegaDetector
DEFAULT_DETECTOR_LABEL_MAP = {
    '1': 'animal',
    '2': 'person',
    '3': 'vehicle'  # available in megadetector v4+
}

# Each version of the detector is associated with some "typical" values
# that are included in output files, so that downstream applications can 
# use them as defaults.
DETECTOR_METADATA = {
    'v2.0.0':
        {'megadetector_version':'v2.0.0',
         'typical_detection_threshold':0.8,
         'conservative_detection_threshold':0.3},
    'v3.0.0':
        {'megadetector_version':'v3.0.0',
         'typical_detection_threshold':0.8,
         'conservative_detection_threshold':0.3},
    'v4.1.0':
        {'megadetector_version':'v4.1.0',
         'typical_detection_threshold':0.8,
         'conservative_detection_threshold':0.3},
    'v5a.0.0':
        {'megadetector_version':'v5a.0.0',
         'typical_detection_threshold':0.2,
         'conservative_detection_threshold':0.05},
    'v5b.0.0':
        {'megadetector_version':'v5b.0.0',
         'typical_detection_threshold':0.2,
         'conservative_detection_threshold':0.05}    
}

DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = DETECTOR_METADATA['v5b.0.0']['typical_detection_threshold']
DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.005  # to include in the output json file


#%% Classes

class ImagePathUtils:
    """A collection of utility functions supporting this stand-alone script"""

    # Stick this into filenames before the extension for the rendered result
    DETECTION_FILENAME_INSERT = '_detections'

    image_extensions = ['.jpg', '.jpeg', '.gif', '.png']

    @staticmethod
    def is_image_file(s):
        """
        Check a file's extension against a hard-coded set of image file extensions
        """
        ext = os.path.splitext(s)[1]
        return ext.lower() in ImagePathUtils.image_extensions

    @staticmethod
    def find_image_files(strings):
        """
        Given a list of strings that are potentially image file names, look for strings
        that actually look like image file names (based on extension).
        """
        return [s for s in strings if ImagePathUtils.is_image_file(s)]

    @staticmethod
    def find_images(dir_name, recursive=False):
        """
        Find all files in a directory that look like image file names
        """
        if recursive:
            strings = glob.glob(os.path.join(dir_name, '**', '*.*'), recursive=True)
        else:
            strings = glob.glob(os.path.join(dir_name, '*.*'))

        image_strings = ImagePathUtils.find_image_files(strings)

        return image_strings


#%% Utility functions

def get_detector_metadata_from_version_string(detector_version):
    """
    Given a MegaDetector version string (e.g. "v4.1.0"), return the metadata for
    the model.  Used for writing standard defaults to batch output files.
    """
    if detector_version not in DETECTOR_METADATA:
        print('Warning: no metadata for unknown detector version {}'.format(detector_version))
        return None
    else:
        return DETECTOR_METADATA[detector_version]


def get_detector_version_from_filename(detector_filename):
    """
    Get the version number component of the detector from the model filename.  
    
    *detector_filename* will almost always end with one of the following:
        
    megadetector_v2.pb
    megadetector_v3.pb
    megadetector_v4.1 (not produed by run_detector_batch.py, only found in Azure Batch API output files)
    md_v4.1.0.pb
    md_v5a.0.0.pt
    md_v5b.0.0.pt
    
    ...for which we identify the version number as "v2.0.0", "v3.0.0", "v4.1.0", 
    "v4.1.0", "v5a.0.0", and "v5b.0.0", respectively.
    """
    fn = os.path.basename(detector_filename)
    known_model_versions = {'v2':'v2.0.0',
                            'v3':'v3.0.0',
                            'v4.1':'v4.1.0',
                            'v5a.0.0':'v5a.0.0',
                            'v5b.0.0':'v5b.0.0'}
    matches = []
    for s in known_model_versions.keys():
        if s in fn:
            matches.append(s)
    if len(matches) == 0:
        print('Warning: could not determine MegaDetector version for model file {}'.format(detector_filename))
        return 'unknown'
    elif len(matches) > 1:
        print('Warning: multiple MegaDetector versions for model file {}'.format(detector_filename))
        return 'multiple'
    else:
        return known_model_versions[matches[0]]
    
    
def is_gpu_available(model_file):
    """Decide whether a GPU is available, importing PyTorch or TF depending on the extension
    of model_file.  Does not actually load model_file, just uses that to determine how to check 
    for GPU availability."""
    
    if model_file.endswith('.pb'):
        import tensorflow.compat.v1 as tf
        gpu_available = tf.test.is_gpu_available()
        print('TensorFlow version:', tf.__version__)
        print('tf.test.is_gpu_available:', gpu_available)                
        return gpu_available
    elif model_file.endswith('.pt'):
        import torch
        gpu_available = torch.cuda.is_available()
        print('PyTorch reports {} available CUDA devices'.format(torch.cuda.device_count()))
        return gpu_available
    else:
        raise ValueError('Unrecognized model file extension for model {}'.format(model_file))


def load_detector(model_file, force_cpu=False):
    """Load a TF or PT detector, depending on the extension of model_file."""
    
    start_time = time.time()
    if model_file.endswith('.pb'):
        from detection.tf_detector import TFDetector
        if force_cpu:
            raise ValueError('force_cpu option is not currently supported for TF detectors, use CUDA_VISIBLE_DEVICES=-1')
        detector = TFDetector(model_file)
    elif model_file.endswith('.pt'):
        from detection.pytorch_detector import PTDetector
        detector = PTDetector(model_file, force_cpu)
    else:
        raise ValueError('Unrecognized model format: {}'.format(model_file))
    elapsed = time.time() - start_time
    print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))
    return detector


#%% Main function
'''
def load_and_run_detector_(model_file, image_file_names, output_dir,
                          render_confidence_threshold=DEFAULT_RENDERING_CONFIDENCE_THRESHOLD,
                          crop_images=False):
    """Load and run detector on target images, and visualize the results."""
    
    if len(image_file_names) == 0:
        print('Warning: no files available')
        return

    print('GPU available: {}'.format(is_gpu_available(model_file)))
    
    detector = load_detector(model_file)
    
    start_time = time.time()
    if model_file.endswith('.pb'):
        from detection.tf_detector import TFDetector
        detector = TFDetector(model_file)
    elif model_file.endswith('.pt'):
        from detection.pytorch_detector import PTDetector
        detector = PTDetector(model_file)
    elapsed = time.time() - start_time
    print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

    detection_results = []
    time_load = []
    time_infer = []

    # Dictionary mapping output file names to a collision-avoidance count.
    #
    # Since we'll be writing a bunch of files to the same folder, we rename
    # as necessary to avoid collisions.
    output_filename_collision_counts = {}

    def input_file_to_detection_file(fn, crop_index=-1):
        """Creates unique file names for output files.

        This function does 3 things:
        1) If the --crop flag is used, then each input image may produce several output
            crops. For example, if foo.jpg has 3 detections, then this function should
            get called 3 times, with crop_index taking on 0, 1, then 2. Each time, this
            function appends crop_index to the filename, resulting in
                foo_crop00_detections.jpg
                foo_crop01_detections.jpg
                foo_crop02_detections.jpg

        2) If the --recursive flag is used, then the same file (base)name may appear
            multiple times. However, we output into a single flat folder. To avoid
            filename collisions, we prepend an integer prefix to duplicate filenames:
                foo_crop00_detections.jpg
                0000_foo_crop00_detections.jpg
                0001_foo_crop00_detections.jpg

        3) Prepends the output directory:
                out_dir/foo_crop00_detections.jpg

        Args:
            fn: str, filename
            crop_index: int, crop number

        Returns: output file path
        """
        fn = os.path.basename(fn).lower()
        name, ext = os.path.splitext(fn)
        if crop_index >= 0:
            name += '_crop{:0>2d}'.format(crop_index)
        fn = '{}{}{}'.format(name, ImagePathUtils.DETECTION_FILENAME_INSERT, '.jpg')
        if fn in output_filename_collision_counts:
            n_collisions = output_filename_collision_counts[fn]
            fn = '{:0>4d}'.format(n_collisions) + '_' + fn
            output_filename_collision_counts[fn] += 1
        else:
            output_filename_collision_counts[fn] = 0
        fn = os.path.join(output_dir, fn)
        return fn

    for im_file in tqdm(image_file_names):

        try:
            start_time = time.time()

            image = viz_utils.load_image(im_file)

            elapsed = time.time() - start_time
            time_load.append(elapsed)

        except Exception as e:
            print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
            result = {
                'file': im_file,
                'failure': FAILURE_IMAGE_OPEN
            }
            detection_results.append(result)
            continue

        try:
            start_time = time.time()

            result = detector.generate_detections_one_image(image, im_file,
                                                            detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD)
            detection_results.append(result)

            elapsed = time.time() - start_time
            time_infer.append(elapsed)

        except Exception as e:
            print('An error occurred while running the detector on image {}. Exception: {}'.format(im_file, e))
            continue

        try:
            if crop_images:

                images_cropped = viz_utils.crop_image(result['detections'], image)

                for i_crop, cropped_image in enumerate(images_cropped):
                    output_full_path = input_file_to_detection_file(im_file, i_crop)
                    cropped_image.save(output_full_path)

            else:

                # image is modified in place
                viz_utils.render_detection_bounding_boxes(result['detections'], image,
                                                          label_map=DEFAULT_DETECTOR_LABEL_MAP,
                                                          confidence_threshold=render_confidence_threshold)
                output_full_path = input_file_to_detection_file(im_file)
                image.save(output_full_path)

        except Exception as e:
            print('Visualizing results on the image {} failed. Exception: {}'.format(im_file, e))
            continue

    # ...for each image

    ave_time_load = statistics.mean(time_load)
    ave_time_infer = statistics.mean(time_infer)
    if len(time_load) > 1 and len(time_infer) > 1:
        std_dev_time_load = humanfriendly.format_timespan(statistics.stdev(time_load))
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_load = 'not available'
        std_dev_time_infer = 'not available'
    print('On average, for each image,')
    print('- loading took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_load),
                                                    std_dev_time_load))
    print('- inference took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_infer),
                                                      std_dev_time_infer))
'''


def load_and_run_detector(model_file, image_file_names, image_labels, novelty_type, output_dir,
                          render_confidence_threshold=DEFAULT_RENDERING_CONFIDENCE_THRESHOLD,
                          crop_images=False):
    """Load and run detector on target images, and visualize the results."""
    
    if len(image_file_names) == 0:
        print('Warning: no files available')
        return

    print('GPU available: {}'.format(is_gpu_available(model_file)))
    
    detector = load_detector(model_file)
    
    start_time = time.time()
    if model_file.endswith('.pb'):
        from detection.tf_detector import TFDetector
        detector = TFDetector(model_file)
    elif model_file.endswith('.pt'):
        from detection.pytorch_detector import PTDetector
        detector = PTDetector(model_file)
    elapsed = time.time() - start_time
    print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

    detection_results = []
    time_load = []
    time_infer = []

    # Dictionary mapping output file names to a collision-avoidance count.
    #
    # Since we'll be writing a bunch of files to the same folder, we rename
    # as necessary to avoid collisions.
    output_filename_collision_counts = {}

    def input_file_to_detection_file(fn, crop_index=-1):
        """Creates unique file names for output files.

        This function does 3 things:
        1) If the --crop flag is used, then each input image may produce several output
            crops. For example, if foo.jpg has 3 detections, then this function should
            get called 3 times, with crop_index taking on 0, 1, then 2. Each time, this
            function appends crop_index to the filename, resulting in
                foo_crop00_detections.jpg
                foo_crop01_detections.jpg
                foo_crop02_detections.jpg

        2) If the --recursive flag is used, then the same file (base)name may appear
            multiple times. However, we output into a single flat folder. To avoid
            filename collisions, we prepend an integer prefix to duplicate filenames:
                foo_crop00_detections.jpg
                0000_foo_crop00_detections.jpg
                0001_foo_crop00_detections.jpg

        3) Prepends the output directory:
                out_dir/foo_crop00_detections.jpg

        Args:
            fn: str, filename
            crop_index: int, crop number

        Returns: output file path
        """
        # fn = os.path.basename(fn).lower()
        fn = os.path.basename(fn)
        name, ext = os.path.splitext(fn)

        # Modify output dir to save detection files in separate folder
        output_parent_dir_list, dirname = output_dir.split('/')[:-1], output_dir.split('/')[-1]
        output_parent_dir_list += [dirname + '_detection']
        new_output_dir = '/'.join(output_parent_dir_list)
        Path(new_output_dir).mkdir(parents=True, exist_ok=True)
        
        if crop_index >= 0:
            name += '_crop{:0>2d}'.format(crop_index)
        fn = '{}{}{}'.format(name, ImagePathUtils.DETECTION_FILENAME_INSERT, '.jpg')
        if fn in output_filename_collision_counts:
            n_collisions = output_filename_collision_counts[fn]
            fn = '{:0>4d}'.format(n_collisions) + '_' + fn
            output_filename_collision_counts[fn] += 1
        else:
            output_filename_collision_counts[fn] = 0
        # fn = os.path.join(output_dir, fn)
        fn = os.path.join(new_output_dir, fn)
        return fn

    output_cols = ['capture_id', 'image_path_rel', 'filename', 'question__count_median', 'question__count_max']
    # if novelty_type == 'type4':
    #     output_cols.append('question__count_max')

    def check_detection_n_grd_truth(d_count, gt_count):
        '''
        Check whether the detector count matches the ground truth count label
        '''
        if gt_count == '51+':
            if d_count >= 51:
                return True
            return False
        elif gt_count == '11-50':
            if 10 < d_count < 51:
                return True
            return False
        elif int(gt_count) == d_count:
            return True
        else:
            return False

    selected_imgs, discarded_imgs = pd.DataFrame(), pd.DataFrame()
    selected_imgs_bboxes = {}

    for im_file in tqdm(image_file_names):

        try:
            start_time = time.time()

            image = viz_utils.load_image(im_file)

            elapsed = time.time() - start_time
            time_load.append(elapsed)

        except Exception as e:
            print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
            result = {
                'file': im_file,
                'failure': FAILURE_IMAGE_OPEN
            }
            detection_results.append(result)
            continue

        try:
            start_time = time.time()

            # result = detector.generate_detections_one_image(image, im_file, detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD)

            result = detector.generate_detections_one_image(image, im_file, detection_threshold=render_confidence_threshold)

            detection_results.append(result)

            # compare the number of detected animals with the ground truth label count
            fname = im_file.split('/')[-1]
            
            if novelty_type == 'type4':
                grd_truth_median_count = image_labels['sum_counts_median_to_bin'].loc[image_labels['filename'] == fname].values[0]
                grd_truth_max_count = image_labels['sum_counts_max_to_bin'].loc[image_labels['filename'] == fname].values[0]
            else:
                grd_truth_median_count = image_labels['question__count_median'].loc[image_labels['filename'] == fname].values[0]
                grd_truth_max_count = image_labels['question__count_max'].loc[image_labels['filename'] == fname].values[0]

            detector_count = len(result['detections'])

            # check if the number of detection matches either the ground truth max_count or median_count
            if check_detection_n_grd_truth(detector_count, grd_truth_median_count) or check_detection_n_grd_truth(detector_count, grd_truth_max_count):
                img_detection = pd.DataFrame(
                        [[image_labels[col].loc[image_labels['filename'] == fname].values[0] for col in output_cols] + [detector_count]],
                        columns=output_cols + ['detect_count']
                        )
                        
                if len(selected_imgs) == 0:
                    selected_imgs = deepcopy(img_detection)
                else:
                    selected_imgs = pd.concat([selected_imgs, img_detection], axis=0,ignore_index=True)


                #***** save the detected bounding boxes in the image
                # selected_imgs_bboxes[fname] = result['detections']
                # selected_imgs_bboxes[fname] = [x['bbox'] for x in result['detections']]
                # selected_imgs_bboxes[fname]['bboxes'] = [x['bbox'] for x in result['detections']]

                selected_imgs_bboxes[fname] = {'bboxes': [x['bbox'] for x in result['detections']]}
                '''
                selected_imgs_bboxes[fname]['image_width'] = 
                selected_imgs_bboxes[fname]['image_height'] = 
                '''

            else:
                img_detection = pd.DataFrame(
                        [[image_labels[col].loc[image_labels['filename'] == fname].values[0] for col in output_cols] + [detector_count]],
                        columns=output_cols + ['detect_count']
                        )
                if len(discarded_imgs) == 0:
                    discarded_imgs = deepcopy(img_detection)
                else:
                    discarded_imgs = pd.concat([discarded_imgs, img_detection], axis=0,ignore_index=True)

            elapsed = time.time() - start_time
            time_infer.append(elapsed)

        except Exception as e:
            print('An error occurred while running the detector on image {}. \nException: {}'.format(im_file, e))
            continue

        '''
        try:
            if crop_images:

                images_cropped = viz_utils.crop_image(result['detections'], image)

                for i_crop, cropped_image in enumerate(images_cropped):
                    output_full_path = input_file_to_detection_file(im_file, i_crop)
                    cropped_image.save(output_full_path)

            else:

                # image is modified in place
                viz_utils.render_detection_bounding_boxes(result['detections'], image,
                                                          label_map=DEFAULT_DETECTOR_LABEL_MAP,
                                                          confidence_threshold=render_confidence_threshold)
                output_full_path = input_file_to_detection_file(im_file)
                image.save(output_full_path)

        except Exception as e:
            print('Visualizing results on the image {} failed. Exception: {}'.format(im_file, e))
            continue
        '''
    # ...for each image

    # get parent dir of output_dir
    label_output_dir_list = output_dir.split('/')[:-1]
    label_output_dir = '/'.join(label_output_dir_list)  
    if novelty_type == 'prenovelty':
        # selected_imgs.to_csv(label_output_dir+f'/Refined_Dataset/prenovelty/correct_images.csv')
        # discarded_imgs.to_csv(label_output_dir+f'/Refined_Dataset/prenovelty/discarded_images.csv')
        selected_imgs.to_csv(label_output_dir+f'/Detector_Refined_Labels/prenovelty_correct_images.csv')
        discarded_imgs.to_csv(label_output_dir+f'/Detector_Refined_Labels/prenovelty_discarded_images.csv')

        with open(label_output_dir+f'/Detector_Refined_Labels/bboxes/prenovelty_bboxes.json', 'w') as fp:
            json.dump(selected_imgs_bboxes, fp)
    else:
        
        # selected_imgs.to_csv(label_output_dir+f'/Refined_Dataset/{novelty_type}_novelty/correct_images.csv')
        # discarded_imgs.to_csv(label_output_dir+f'/Refined_Dataset/{novelty_type}_novelty/discarded_images.csv')
        selected_imgs.to_csv(label_output_dir+f'/Detector_Refined_Labels/{novelty_type}_correct_images.csv')
        discarded_imgs.to_csv(label_output_dir+f'/Detector_Refined_Labels/{novelty_type}_discarded_images.csv')
        
        # with open(label_output_dir+f'/Detector_Refined_Labels/bboxes/{novelty_type}_novelty_bboxes.json', 'w') as fp:
        #     json.dump(selected_imgs_bboxes, fp)

        with open(label_output_dir+f'/bboxes/{novelty_type}_novelty_bboxes.json', 'w') as fp:
            json.dump(selected_imgs_bboxes, fp)

    ave_time_load = statistics.mean(time_load)
    ave_time_infer = statistics.mean(time_infer)
    if len(time_load) > 1 and len(time_infer) > 1:
        std_dev_time_load = humanfriendly.format_timespan(statistics.stdev(time_load))
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_load = 'not available'
        std_dev_time_infer = 'not available'
    print('On average, for each image,')
    print('- loading took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_load),
                                                    std_dev_time_load))
    print('- inference took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_infer),
                                                      std_dev_time_infer))


#%% Command-line driver
'''
def main0():

    parser = argparse.ArgumentParser(
        description='Module to run an animal detection model on images')
    parser.add_argument(
        'detector_file',
        help='Path to TensorFlow (.pb) or PyTorch (.pt) detector model file')
    group = parser.add_mutually_exclusive_group(required=True)  # must specify either an image file or a directory
    group.add_argument(
        '--image_file',
        help='Single file to process, mutually exclusive with --image_dir')
    group.add_argument(
        '--image_dir',
        help='Directory to search for images, with optional recursion by adding --recursive')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurse into directories, only meaningful if using --image_dir')
    parser.add_argument(
        '--output_dir',
        help='Directory for output images (defaults to same as input)')
    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_RENDERING_CONFIDENCE_THRESHOLD,
        help=('Confidence threshold between 0 and 1.0; only render boxes above this confidence'
              ' (but only boxes above 0.005 confidence will be considered at all)'))
    parser.add_argument(
        '--crop',
        default=False,
        action="store_true",
        help=('If set, produces separate output images for each crop, '
              'rather than adding bounding boxes to the original image'))
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.detector_file), 'detector file {} does not exist'.format(args.detector_file)
    assert 0.0 < args.threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'  # Python chained comparison

    if args.image_file:
        image_file_names = [args.image_file]
    else:
        image_file_names = ImagePathUtils.find_images(args.image_dir, args.recursive)

    print('Running detector on {} images...'.format(len(image_file_names)))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        if args.image_dir:
            args.output_dir = args.image_dir
        else:
            # but for a single image, args.image_dir is also None
            args.output_dir = os.path.dirname(args.image_file)

    load_and_run_detector(model_file=args.detector_file,
                          image_file_names=image_file_names,
                          output_dir=args.output_dir,
                          render_confidence_threshold=args.threshold,
                          crop_images=args.crop)
'''


def main():

    # imgs_dir = '../../../../../../../hpc/share/sail_on3/type4_novelty'

    parser = argparse.ArgumentParser(
        description='Module to run an animal detection model on images')
    parser.add_argument(
        'detector_file',
        help='Path to TensorFlow (.pb) or PyTorch (.pt) detector model file')
    
    group = parser.add_mutually_exclusive_group(required=True)  # must specify either an image file or a directory
    group.add_argument(
        '--image_file',
        help='Single file to process, mutually exclusive with --image_dir'
        )
    group.add_argument(
        '--image_dir',
        help='Directory to search for images, with optional recursion by adding --recursive'
        )
    parser.add_argument(
        '--image_metadata',
        help='Single file to process, mutually exclusive with --image_metadata'
        )
    parser.add_argument(
        '--novelty_type',
        help='Type of novelty being processed --novelty_type'
        )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurse into directories, only meaningful if using --image_dir'
        )
    parser.add_argument(
        '--output_dir',
        help='Directory for output images (defaults to same as input)'
        )
    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_RENDERING_CONFIDENCE_THRESHOLD,
        help=('Confidence threshold between 0 and 1.0; only render boxes above this confidence'
              ' (but only boxes above 0.005 confidence will be considered at all)')
              )
    parser.add_argument(
        '--crop',
        default=False,
        action="store_true",
        help=('If set, produces separate output images for each crop, '
              'rather than adding bounding boxes to the original image')
              )
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.detector_file), 'detector file {} does not exist'.format(args.detector_file)
    assert 0.0 < args.threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'  # Python chained comparison

    if args.image_file:
        image_file_names = [args.image_file]
    else:
        print(f'current working dir: {os.getcwd()}')
        print(f'image dir being used: {args.image_dir}')
        image_file_names = ImagePathUtils.find_images(args.image_dir, args.recursive)

    print('Running detector on {} images...'.format(len(image_file_names)))

    if args.image_metadata:
        image_labels = pd.read_csv(args.image_metadata)

        def remove_subdir(x):
            return x.split('/')[-1]

        # print(f'image labels columns: {image_labels.columns}')
        image_labels['filename'] = image_labels['image_path_rel'].apply(remove_subdir)

        # print(f'*** Head of metadata filename: {image_labels["filename"].head(20)}')

    else:
        raise ValueError(f'\'--image_labels\' argument not provided, please provide .csv images label!')

    if args.novelty_type:
        assert args.novelty_type in ['prenovelty', 'type2', 'type3', 'type4', 'type6']
        novelty = args.novelty_type

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        if args.image_dir:
            args.output_dir = args.image_dir
        else:
            # but for a single image, args.image_dir is also None
            args.output_dir = os.path.dirname(args.image_file)

    load_and_run_detector(model_file=args.detector_file,
                          image_file_names=image_file_names,
                          image_labels = image_labels,
                          novelty_type=novelty,
                          output_dir=args.output_dir,
                          render_confidence_threshold=args.threshold,
                          crop_images=args.crop)


if __name__ == '__main__':
    # ***** shell command
    # python detection/run_detector.py "../md_v5a.0.0.pt" --image_dir "../../../../../sail_on3/Images/type4_novelty" \
    # --image_metadata "../../../../../sail_on3/candidate_subset/type4_labels_summed.csv" --novelty_type "type4" --threshold 0.18
    main()


#%% Interactive driver

if False:

    #%%
    model_file = r'c:\temp\models\md_v4.1.0.pb'
    image_file_names = ImagePathUtils.find_images(r'c:\temp\demo_images\ssverymini')
    output_dir = r'c:\temp\demo_images\ssverymini'
    render_confidence_threshold = 0.8
    crop_images = True

    load_and_run_detector(model_file=model_file,
                          image_file_names=image_file_names,
                          output_dir=output_dir,
                          render_confidence_threshold=render_confidence_threshold,
                          crop_images=crop_images)
