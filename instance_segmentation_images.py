from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

CWD_PATH = Path('.')
TF_MODELS_PATH = Path('../TensorFlow Object Detection Models/trained_models')
# Path to frozen detection graph. This is the actual model that is used for the object detection
MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
PATH_TO_CKPT = TF_MODELS_PATH / MODEL_NAME / 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box
PATH_TO_LABELS = CWD_PATH / 'object_detection' / 'data' / 'mscoco_label_map.pbtxt'

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(str(PATH_TO_LABELS))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def model_load_into_memory():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(str(PATH_TO_CKPT), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    return detection_graph

def run_inference_for_single_image(image, sess, graph):
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 
                'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
        
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})
    
    # All outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0].astype(np.float32)
            
    return output_dict
    
def visualize_results(image, output_dict):
    """Returns the resulting image after being passed to the model.
    
    Args:
        image (ndarray): Original image given to the model.
        output_dict (dict): Dictionary with all the information provided by the model.
    
    Returns:
        image (ndarray): Visualization of the results form above.
        
    """
    vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4)
    
    return image


def main():
    detection_graph = model_load_into_memory()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            images = Path('images').glob('**/*.jpg')
            for image in images:
                img = cv2.imread(str(image))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                output = run_inference_for_single_image(img, sess, detection_graph)
                processed_image = visualize_results(img, output)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                Path('segmented_images').mkdir(exist_ok=True)
                cv2.imwrite('segmented_images/{}'.format(str(image.name)), processed_image)
                
    print("Images segmetned correctly: Check the 'segmented_images/' folder to see the results")

if __name__ == '__main__':
    main()
    