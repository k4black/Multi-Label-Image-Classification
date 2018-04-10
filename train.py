"""This project uses a pre-trained network for ImageNet, adding a new layer that
will be learned for new labels, and displays a resume in TensorBoard.


DESCRIPTION:
This model take a pre-trained Inception v3 shape graph (trained for the ImageNet
images) and replaces the top layer of the model with new fully-connected
layer that can recognize other classes of images (multi-label classification).

The former top layer (node by name - 'softmax') receives as input a
2048-dimensional vector. And that model create a new layer instead, which
receives as the input this vector and has the sigmoid activation function that
enables multi-label classification.

This model base on 'Retrain an Image Classifier for New Categories' TensorFlow
tutorial (www.tensorflow.org/tutorials/image_retraining), but there are
significant changes. First of all - single-label classification (softmax layer)
was changed to multi-label classification (sigmoid layer), as well as all system
of caching and loading images for learning was rebuild for better optimisation.


PREPARATIONS:
Before training the model, you should to prepare some file structure and some
separate files. The absence of these preliminary actions will entail a program
error - access-error or faulty output model.
(There is no verification of the correctness of the input data, it is left for
lightness of the source code)

The LABELS_DIR/labels.csv file describing all the available labels in the
certain order in which they will be described during the training. This is
described in the csv file format, for example:
    label_1,label_2, ... ,label_N

The training images folder (IMAGE_DIR) should have a structure like this:
    ./images/train/image_name_1.jpg
    ./images/train/image_name_2.jpg
    ...
    ./images/train/image_name_N.jpg

Also there should be a LABELS_DIR/train.csv file describing (giving the correct
labels for each image) images from the IMAGE_DIR directory in the csv format.
For instance, the file may look like this:
    image_name_1.jpg,0,1,0,0, ... ,0,1,1
    image_name_2.jpg,0,0,1,0, ... ,1,0,0
    ...
    image_name_N.jpg,1,0,0,0, ... ,0,0,0


OUTPUT:
This script produces a new model file (tensorflow graph) that can be loaded and
run by any TensorFlow-compatibility program, for example the label.py or OpenCV.


TENSORBOARD:
By default, this script will log summaries in the format for TensorBoard to
SUMMARIES_DIR (./tmp/logs by default) directory.
Visualize the summaries with this command:
    $ tensorboard --logdir ./tmp/logs


REFERENCE:
This network was used for the image multi-labeling at the competition by Intel
(www.kaggle.com/c/delta9). The initial data set is a 36000 images described by
17 categories. It is required to mark 4000 more images.

This architecture allowed me to achieve the score at 0.962 without significant
time expenditure for learning the whole model.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import hashlib
import os
import random
import re
import sys
import tarfile
import csv

import numpy as np
from PIL import Image
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


# === Input and output. === #

IMAGE_DIR = './images/train'  # Path to folders of labeled images.
LABELS_DIR = './labels'  # Path to folders with labels for images.

SUMMARIES_DIR = './tmp/logs'  # Path to save summary logs for TensorBoard.
BACKUP_DIR = './tmp/backups'  # Path to save backups graphs if any.
CHECKPOINT_DIR = './tmp/checkpoints'  # Path to variable checkpoints.

MODEL_DIR = './tmp/imagenet'  # Path to pre-trained model graph.
BOTTLENECK_DIR = './tmp/bottleneck'  # Path to cache bottleneck directory.

OUTPUT_GRAPH = './tmp/output_graph.pb'  # Path to save the trained graph.
FINAL_TENSOR_NAME = 'result'  # The name of the output classification layer.


# === Hyper-parameters of the training configuration. === #

TRAINING_STEPS = 1000  # How many training steps to run. (IT IS NOT EPOCH!)
LEARNING_RATE = 0.1  # 'Speed' of training training.
MOMENTUM = 0.1  # For Momentum-SGD.

TESTING_PERCENTAGE = 10  # Percentage of images to use as a 'test' set.
VALIDATION_PERCENTAGE = 10  # Percentage of images to use as a 'validation' set.

EVAL_STEP_INTERVAL = 50  # How often to evaluate the training results.
CHECKPOINT_STEP_INTERVAL = 500  # If any.

TRAIN_BATCH_SIZE = 150  # How many images to train on at a one step.
TEST_BATCH_SIZE = 4000  # Assay the accuracy of the model after all training.
VALIDATION_BATCH_SIZE = 500  # Assay the accuracy of the model during training.


# === Inception v3 constants === #

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
MODEL_FILE_NAME = 'classify_image_graph_def.pb'


def prepare_file_system():
    """Makes sure the required folders exists on disk or create them.

    Return:
        Nothing

    """

    if tf.gfile.Exists(SUMMARIES_DIR):
        tf.gfile.DeleteRecursively(SUMMARIES_DIR)
    tf.gfile.MakeDirs(SUMMARIES_DIR)

    if tf.gfile.Exists(CHECKPOINT_DIR):
        tf.gfile.DeleteRecursively(CHECKPOINT_DIR)
    tf.gfile.MakeDirs(CHECKPOINT_DIR)

    if tf.gfile.Exists(BACKUP_DIR):
        tf.gfile.DeleteRecursively(BACKUP_DIR)
    tf.gfile.MakeDirs(BACKUP_DIR)


def maybe_download_and_extract():
    """Download and extract model .tar file.
    If the pre-trained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.

    Note:
        _progress(): A function that displays progress of downloading.

    Return:
        Nothing

    """

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    filename = DATA_URL.split('/')[-1]
    filepath = MODEL_DIR + '/' + filename

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading {} {:.1f}%'.format(
                filename, float(count * block_size) / float(total_size) * 100.0) )
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()

        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)


def create_label_list(filename):
    """Builds a list of labels for images from the .csv file.

    Args:
        filename: String name of .csv file containing list of labels for images.
            All labels should be placed on a one line with a ',' between them

    Note:
        LABELS_DIR: Path to folders with labels for images.

    Returns:
        A list containing an string labels for each class of images.

    """

    # get the names of the labels
    labels_name = []
    with open(LABELS_DIR + '/' + filename, newline='') as labels_file:
        reader = csv.reader(labels_file)

        for row in reader:  # Get only first line
            labels_name = [i for i in row]
            break

    return labels_name


def create_image_lists(filename, testing_percentage, validation_percentage):
    """Create a dictionary of training images from the .csv file.
    Splits images into stable 'training', 'testing', and 'validation' sets, and
    returns a data structure describing the images.
    (The images distribution depends on the file name only --> at any starts the
    same picture will distributed into the same set)

    Args:
        filename: String name of .csv file containing list of labels for images.
            All files description should be placed on a separate line
            in the format: [name],[probability#1],..,[probability#n]
        testing_percentage: Integer percentage of images used for the test
        validation_percentage: Integer percentage of images used for the
            validation.

    Note:
        LABELS_DIR: Path to folders with labels for images.

    Returns:
        A dictionary containing an entry for each image, with images split
        into training, testing, and validation sets within each label.
        result = {
            'training': training_images{ 'name': [0.0,1.0,0.0,1.0...], ...},
            'testing': testing_images{ 'name': [0.0,1.0,0.0,1.0...], ...},
            'validation': validation_images{ 'name': [0.0,1.0,0.0,1.0...], ...}
        }

    """

    # Preparing dictionary of the sets
    result = {set_class: {} for set_class in ['validation', 'testing', 'training']}

    with open(LABELS_DIR + '/' + filename, newline='') as labels_file:
        reader = csv.reader(labels_file)

        for row in reader:  # for each line of .csv row=(filename.jpg,0,0,1,0...)

            # Get labels list for that image.
            labels_count = len(row)-1
            if labels_count <= 1:
                continue

            labels_for_file = np.zeros(labels_count, dtype=np.float32)
            for i in range(1, labels_count+1):  # for each label (0,0,1,0...)
                if row[i] == '1':
                    labels_for_file[i-1] = 1.0


            # Distribute this image in one of the sets.
            base_name = row[0]  # File name (first column)

            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a
            # way of grouping photos that are close variations of each other.
            # For example this is used in the plant disease data set to group
            # multiple pictures of the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', row[0])

            # This looks a bit magical, but we need to decide whether this file
            # should go into the 'training', 'testing', or 'validation' sets,
            # and we want to keep existing files in the same set even if more
            # files are subsequently added.
            # To do that, we need a stable way of deciding based on just the
            # file name, so we do a hash of that and then use that to generate
            # a probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))

            tmp = {}  # Dictionary for the current set class
            if percentage_hash < validation_percentage:
                tmp = result['validation']

            elif percentage_hash < (testing_percentage + validation_percentage):
                tmp = result['testing']

            else:
                tmp = result['training']

            # Add element into current set
            dict.setdefault(tmp, base_name, labels_for_file)

    return result


def get_class_distribution(image_lists, labels):
    class_num = [0 for _ in labels]

    for _, image_set in dict.items(image_lists):
        for image, classes in dict.items(image_set):  # Get only first line
            # print(image, '!', classes)
            for i, item in enumerate(classes):
                if item == 1.0:
                    class_num[i] += 1

    print('Classes distribution: ', class_num)

    return class_num


def create_model_graph():
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Note:
        MODEL_DIR, MODEL_FILE_NAME: Directory and name of pre-trained graph file.

    Returns:
        Graph holding the pre-trained Inception v3 network, and various tensors
        we'll be manipulating.

    """

    with tf.Session() as sess:
        model_filename = MODEL_DIR + '/' + MODEL_FILE_NAME

        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME, RESIZED_INPUT_TENSOR_NAME]))

    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.
    Feeds the image of an already trained network and save output
    2048-dimensions tensor of penultimate layer into the file. It also deals
    with the transfer of the image to the right size.

    Args:
        sess: Current active TensorFlow Session.
        image_data: String of raw JPEG data.
        image_data_tensor: Input data layer in the graph.
        bottleneck_tensor: Layer before the final layer 'softmax'.

    Returns:
        Numpy array of bottleneck values.

    """

    bottleneck_values = sess.run( bottleneck_tensor, {image_data_tensor: image_data} )
    bottleneck_values = np.squeeze(bottleneck_values)

    return bottleneck_values


def create_bottleneck(sess, image_name, jpeg_data_tensor, bottleneck_tensor):
    """Calculates bottleneck values for an image.
    If a cached version of the bottleneck data exists on-disk, do nothing,
    otherwise calculate the data and save it to disk for future use.

    Args:
        sess: The current active TensorFlow Session.
        available number of images for the label, so it can be arbitrarily large.
        image_name: String name of current image
        jpeg_data_tensor: The tensor to feed loaded jpeg data into.
        bottleneck_tensor: The output tensor for the bottleneck values.

    Returns:
        Nothing

    """

    bottleneck_path = BOTTLENECK_DIR + '/' + image_name + '.npy'

    if not os.path.exists(bottleneck_path):
        # print("Creating bottleneck at '" + bottleneck_path + "'")

        image_path = IMAGE_DIR + '/' + image_name
        image_data = gfile.FastGFile(image_path, 'rb').read()

        bottleneck_values = run_bottleneck_on_image(sess, image_data,
                                                    jpeg_data_tensor, bottleneck_tensor)

        np.save(bottleneck_path, bottleneck_values)


def create_distorted_bottleneck(sess, image_name, distorted_image_name, jpeg_data_tensor,
                                bottleneck_tensor):
    """Calculates bottleneck values for an image.
    If a cached version of the bottleneck data exists on-disk, do nothing,
    otherwise calculate the data and save it to disk for future use.

    Args:
        sess: The current active TensorFlow Session.
        available number of images for the label, so it can be arbitrarily large.
        image_name: String name of current image
        distorted_image_name: String name of distorted image.
        jpeg_data_tensor: The tensor to feed loaded jpeg data into.
        bottleneck_tensor: The output tensor for the bottleneck values.

    Returns:
        Nothing

    """

    bottleneck_path = BOTTLENECK_DIR + '/' + distorted_image_name + '.npy'

    if not os.path.exists(bottleneck_path):

        image_path = IMAGE_DIR + '/' + image_name

        im = Image.open(image_path)
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im.save("./tmp/tmp.jpg")
        image_data = gfile.FastGFile('./tmp/tmp.jpg', 'rb').read()

        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor,
                                                    bottleneck_tensor)

        np.save(bottleneck_path, bottleneck_values)  # Save as std numpy array


def cache_bottlenecks(sess, image_lists, jpeg_data_tensor, bottleneck_tensor,
                      create_distorted=False, class_num=None):
    """Ensures all bottlenecks for the 'training', 'testing', and 'validation'
    sets are created and cached.
    Because we're likely to read the same image multiple times it can speed
    things up a lot if we calculate the bottleneck layer values once for each
    image during pre-processing, and then just read those cached values
    repeatedly during training. Here we go through all the images we've found,
    calculate those values, and save them off. Also bottleneck values
    calculating for discorded images (the images themselves are not saved).

    Args:
        sess: The current active TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        jpeg_data_tensor: Input tensor for jpeg data from file.
        bottleneck_tensor: The penultimate output layer of the graph.
        create_distorted: Distort image to increase training sample

    Note:
        BOTTLENECK_DIR: Folder string holding cached files of bottleneck values.
        _progress(): A function that displays progress of bottleneck creation.

    Returns:
        Nothing

    """

    def _progress(set_name, count, total):
        sys.stdout.write("\r>> Creating '{}' set {:.1f}% [{:d}/{:d}]".format(
            set_name, float(count+1) / float(total) * 100.0, count+1, total) )
        sys.stdout.flush()


    # For each image from each set calculate it's bottleneck
    how_many_bottlenecks = 0
    for category, category_dict in image_lists.items():

        num_of_images = len(category_dict.keys())

        for image_index, image_name in enumerate(category_dict.keys()):
            create_bottleneck(sess, image_name, jpeg_data_tensor, bottleneck_tensor)

            how_many_bottlenecks += 1
            _progress(category, image_index, num_of_images)
        print()


    # For each image from training set (ONLY!) create a distorted copy
    if create_distorted:
        distorted_dict = {}  # temp dict

        category_dict = image_lists['training']
        num_of_images = len(category_dict.keys())

        if class_num is not None:
            print('Normalisation...')
            class_min = list.copy(class_num)

            for i, item in enumerate(class_num):
                if item - (sum(class_num) / len(class_num) / 3) < 0:
                    class_min[i] = 1.0
                else:
                    class_min[i] = 0.0

            for image_index, image_name in enumerate(category_dict.keys()):
                labels_for_file = category_dict[image_name]  # Get a ground_truth vector

                equal_labels = 0
                for i in range(len(class_min)):
                    if class_min[i] == labels_for_file[i] == 1.0:
                        equal_labels += 1
                        break

                if equal_labels > 0:
                    distorted_image_name = image_name + '_ds'

                    # Add a new bottleneck name into temporary dict
                    dict.setdefault(distorted_dict, distorted_image_name, labels_for_file)

                    create_distorted_bottleneck(sess, image_name, distorted_image_name,
                                                jpeg_data_tensor, bottleneck_tensor)

                    how_many_bottlenecks += 1
                _progress('distorted', image_index, num_of_images)

        else:
            for image_index, image_name in enumerate(category_dict.keys()):
                labels_for_file = category_dict[image_name]  # Get a ground_truth vector
                distorted_image_name = image_name + '_ds'

                # Add a new bottleneck name into temporary dict
                dict.setdefault(distorted_dict, distorted_image_name, labels_for_file)

                create_distorted_bottleneck(sess, image_name, distorted_image_name,
                                            jpeg_data_tensor, bottleneck_tensor)

                how_many_bottlenecks += 1
                _progress('distorted', image_index, num_of_images)
        print()

        dict.update(category_dict, distorted_dict)


    print('Finished: ' + str(how_many_bottlenecks) + ' bottleneck files created.')


def get_bottleneck(image_name):
    """Retrieves bottleneck values for an image from the on-disk.

    Args:
        image_name: String name of current image

    Returns:
        Numpy array of values produced by the bottleneck layer for the image.

    """

    bottleneck_path = BOTTLENECK_DIR + '/' + image_name + '.npy'
    bottleneck_values = np.load(bottleneck_path)

    return bottleneck_values


def get_random_cached_bottlenecks(image_lists, how_many, category):
    """Retrieves bottleneck values for cached images.
    This function retrieve cached bottleneck values directly from disk. It picks
    a random set of images from the specified category.

    Args:
        image_lists: Dictionary of training images for each label.
        how_many: The number of bottleneck values to return.
        category: Name string of which set to pull from - 'training', 'testing',
            or 'validation'.

    Returns:
        List of bottlenecks(numpy arrays) and their corresponding ground truths.

    """

    bottlenecks = []
    ground_truths = []

    class_dict = image_lists[category]
    image_names = list(dict.keys(class_dict))

    for unused_i in range(how_many):
        image_name = random.choice(image_names)

        bottleneck = get_bottleneck(image_name)
        ground_truth = class_dict[image_name]

        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor.
    (for TensorBoard visualization).

    Returns:
        Nothing.

    """

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean' + '/' + name, mean)

        with tf.name_scope('stddev' + '/' + name):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev' + '/' + name, stddev)
        tf.summary.scalar('max' + '/' + name, tf.reduce_max(var))
        tf.summary.scalar('min' + '/' + name, tf.reduce_min(var))
        tf.summary.histogram('histogram' + '/' + name, var)


def add_final_training_ops(class_count, bottleneck_tensor):
    """Adds a new sigmoid fully-connected layer for training a new categories.
    We need to train the new top layer to identify our new categories, so this
    function adds the right operations to the graph, along with some variables
    to hold the weights, and sets up all the gradients for the backward pass.

    Args:
        class_count: Integer of how many categories of things we're trying to
        recognize.
        bottleneck_tensor: The output of the main CNN graph.

    Note:
        FINAL_TENSOR_NAME: String name of our new layer that produces results.
        BOTTLENECK_TENSOR_SIZE: Size of tensor which produce penultimate layer.
        LEARNING_RATE, MOMENTUM: Hyper-parameters for the Momentum-SGD.

    Returns:
        The tensors for the training and cross entropy results, and tensors for the
        bottleneck input and ground truth input.

    """

    # Create a score for loading out bottlenecks
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(bottleneck_tensor,
                                                       [None, BOTTLENECK_TENSOR_SIZE],
                                                       name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32, [None, class_count],
                                            name='GroundTruthInput')

    # Organizing the following ops so they are easier to see in TensorBoard.
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights, layer_name + '/weights')

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases, layer_name + '/biases')

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram(layer_name + '/pre_activations', logits)

    final_tensor = tf.nn.sigmoid(logits, name=FINAL_TENSOR_NAME)
    tf.summary.histogram(FINAL_TENSOR_NAME + '/activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.losses.sigmoid_cross_entropy(ground_truth_input, logits)

        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM)
        train_step = optimizer.minimize(cross_entropy_mean)

    return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
        result_tensor: The new final node that produces results.
        ground_truth_tensor: The node we feed ground truth data into.

    Returns:
        Evaluation step

    """

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # The 'result_tensor' provides a 'probability' for each class, so we
            # have to convert this vector into a vector ({0,1}) for comparison
            # with a vector 'ground_truth_tensor'
            prediction = tf.round(result_tensor)
            # Returns the truth value of (x == y) element-wise.
            correct_prediction = tf.equal(prediction, ground_truth_tensor)

        with tf.name_scope('accuracy'):
            # We count the 'accuracy' as: the number of correct assumptions to
            # divide by the number of all assumptions
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', evaluation_step)

    return evaluation_step


def main(_):
    sess = tf.Session()

    # Create necessary directories
    prepare_file_system()


    # Get a list of labels from .csv file
    labels = create_label_list('labels.csv')
    class_count = len(labels)


    # Set up the pre-trained graph.
    maybe_download_and_extract()
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_model_graph())

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(class_count, bottleneck_tensor)

    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)


    # Create a train saver that is used to restore values into an eval graph
    # when exporting models.
    train_saver = tf.train.Saver()


    # Look at the .cvs file and create lists of all the images.
    image_lists = create_image_lists('train.csv', TESTING_PERCENTAGE, VALIDATION_PERCENTAGE)

    # Get labels distribution
    class_num = get_class_distribution(image_lists, labels)

    # We'll make sure we've calculated the 'bottleneck' summaries and cached them on disk.
    cache_bottlenecks(sess, image_lists, jpeg_data_tensor, bottleneck_tensor, create_distorted=True,
                      class_num=class_num)

    class_num = get_class_distribution(image_lists, labels)


    # Merge all the summaries and write them out to SUMMARIES_DIR
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/validation')
    print('Summaries successfully created.')


    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Run the training for as many cycles as requested on the command line.
    for step in range(TRAINING_STEPS):

        # Get a batch of input bottleneck values from the cache stored on disk.
        train_bottlenecks, train_ground_truth = \
            get_random_cached_bottlenecks(image_lists, TRAIN_BATCH_SIZE, 'training')

        # Feed the bottlenecks and ground truth into the graph.
        train_summary, _ = sess.run([merged, train_step], 
                                    feed_dict={bottleneck_input: train_bottlenecks,
                                               ground_truth_input: train_ground_truth})

        # Capture training summaries for TensorBoard with the `merged` op.
        train_writer.add_summary(train_summary, step)

        # Every so often, print out how well the graph is training.
        is_last_step = (step + 1 == TRAINING_STEPS)
        if (step % EVAL_STEP_INTERVAL) == 0 or is_last_step:

            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy], feed_dict={bottleneck_input: train_bottlenecks,
                                                             ground_truth_input: train_ground_truth})

            print('{}: Step {:d}: Train accuracy = {:.1f}%'.format(datetime.now(), step,
                                                                   train_accuracy * 100))
            print('{}: Step {:d}: Cross entropy = {:.6f}'.format(datetime.now(), step,
                                                                 cross_entropy_value))


            # Run a validation step.
            validation_bottlenecks, validation_ground_truth = (
                get_random_cached_bottlenecks(image_lists, VALIDATION_BATCH_SIZE, 'validation'))

            validation_summary, validation_accuracy = sess.run(
                [merged, evaluation_step], feed_dict={bottleneck_input: validation_bottlenecks,
                                                      ground_truth_input: validation_ground_truth})

            # Capture training summaries for TensorBoard with the `merged` op.
            validation_writer.add_summary(validation_summary, step)

            print('{}: Step {:d}: Validation accuracy = {:.1f}%'.format(datetime.now(), step,
                                                                        validation_accuracy * 100))
            print()


            # Save a checkpoint of the train graph, to restore it after.
            if (step % CHECKPOINT_STEP_INTERVAL) == 0:
                train_saver.save(sess, CHECKPOINT_DIR + '/model_' + str(step) + 'e' + '.ckpt')


    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    test_bottlenecks, test_ground_truth = get_random_cached_bottlenecks(image_lists,
                                                                        TEST_BATCH_SIZE, 'testing')
    test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                         ground_truth_input: test_ground_truth})

    print('//==== Final test accuracy = {:.1f}% ====//'.format(test_accuracy * 100))

    # Write out the trained graph and labels with the weights stored as constants.
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(),
                                                                 [FINAL_TENSOR_NAME])

    with gfile.FastGFile(OUTPUT_GRAPH, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


# Run the main(_)
if __name__ == '__main__':
    tf.app.run()
