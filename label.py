import re
import tensorflow as tf
import sys
import os


IMAGE_DIR = './images/test'  # Path to folders of labeled images.
LABELS_DIR = './labels'  # Path to folders with labels for images.

OUTPUT_GRAPH = './tmp/output_graph.pb'  # Where to save the trained graph.
FINAL_TENSOR_NAME = 'result'  # The name of the output classification layer in the retrained graph.
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'



# Loading trained graph from file
with tf.gfile.FastGFile(OUTPUT_GRAPH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


# Get list of images
file_list = []
for _, _, file_names in os.walk(IMAGE_DIR):
    file_list.extend(file_names)
    break

image_list = [f for f in file_list if re.match(r'.*\.jpg', f)]
list.sort(image_list)


def _progress(count, total):
    sys.stdout.write("\r>> Labeling {:.1f}% [{:d}/{:d}]".format(
        float(count) / float(total) * 100.0, count, total))
    sys.stdout.flush()


with tf.Session() as sess, open(LABELS_DIR + '/' + 'test.csv', 'w') as f:
    f.write('image,umbrella,shirt,shorts,skirt,sweater,suit,jeans,gun,jacket,bag,gloves,trousers,tie,dress,hat,scarf,glasses\n')

    for index, image in enumerate( image_list):
        image_path = IMAGE_DIR + '/' + image
        # Read image data as binary file
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        # Feed the image data (jpeg binary) as the input of the graph
        # (this model already have resize node) and get first prediction
        final_tensor = sess.graph.get_tensor_by_name(FINAL_TENSOR_NAME+':0')

        predictions = sess.run(final_tensor, {JPEG_DATA_TENSOR_NAME: image_data})

        f.write('%s' % image)
        for score in predictions[0]:
            f.write(',{:.6f}'.format(score))
        f.write('\n')

        _progress(index, len(image_list))
