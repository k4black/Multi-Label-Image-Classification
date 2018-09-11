# Multi-Label-Image-Classification

This project uses a pre-trained network for ImageNet, adding a new layer that
will be learned for new labels, and displays a resume in TensorBoard.


### DESCRIPTION:
This model take a pre-trained [Inception v3](https://arxiv.org/abs/1512.00567) shape graph (trained for the *ImageNet*
images) and replaces the top layer of the model with new fully-connected
layer that can recognize other classes of images (multi-label classification).
The former top layer (node by name - *'softmax'*) receives as input a
2048-dimensional vector. And that model create a new layer instead, which
receives as the input this vector and has the sigmoid activation function that
enables multi-label classification.
![Inception v3](https://habrastorage.org/files/84d/8a8/0a2/84d8a80a299e440c9f1892485c2e0803.png)
This model base on 'Retrain an Image Classifier for New Categories' [TensorFlow tutorial](https://http://tensorflow.org/tutorials/image_retraining/) , but there are
significant changes. First of all - single-label classification (*softmax* layer)
was changed to multi-label classification (*sigmoid* layer), as well as all system
of caching and loading images for learning was rebuild for better optimisation.

### PREPARATIONS:
Before training the model, you should to prepare some file structure and some
separate files. The absence of these preliminary actions will entail a program
error - access-error or faulty output model.
You can download [dataset](https://www.kaggle.com/c/delta9/data) wich I used for this project.
*(There is **no** verification of the correctness of the input data, it is left for
lightness of the source code)*
The `LABELS_DIR/labels.csv` file describing all the available labels in the
certain order in which they will be described during the training. This is
described in the csv file format, for example:
    
    label_1,label_2, ... ,label_N
    
The training images folder (`IMAGE_DIR`) should have a structure like this:
    
    ./images/train/image_name_1.jpg
    ./images/train/image_name_2.jpg
    ...
    ./images/train/image_name_N.jpg
    
    
Also there should be a `LABELS_DIR/train.csv` file describing (giving the correct
labels for each image) images from the `IMAGE_DIR` directory in the csv format.
For instance, the file may look like this:
    
    image_name_1.jpg,0,1,0,0, ... ,0,1,1
    image_name_2.jpg,0,0,1,0, ... ,1,0,0
    ...
    image_name_N.jpg,1,0,0,0, ... ,0,0,0
    
    
### OUTPUT:
This script produces a new model file (tensorflow graph) that can be loaded and
run by any TensorFlow-compatibility program, for example the *label.py* or *OpenCV*.

### TENSORBOARD:
By default, this script will log summaries in the format for TensorBoard to
`SUMMARIES_DIR (./tmp/logs by default)` directory.
Visualize the summaries with this command:

    $ tensorboard --logdir ./tmp/logs
    
### REFERENCE:
This network was used for the image multi-labeling at the [competition by Intel](www.kaggle.com/c/delta9). The initial data set is a 36000 images described by
17 categories. It is required to mark 4000 more images.
This architecture allowed me to achieve the score at 0.962 without significant
time expenditure for learning the whole model.
