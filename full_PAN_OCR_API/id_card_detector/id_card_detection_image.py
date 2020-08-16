
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
import re
from tesserect_ocr import preprocess_image, get_text
import pytesseract


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

CWD = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join("id_card_detector/model",'frozen_inference_graph.pb')

# Path to label map file
#PATH_TO_LABELS = os.path.join("id_card_detector/",'data','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph, config=config)

# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
# Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
def crop_document(image):
    boxed_image, cropped_image, all_boxes = [],[], []
    try:
        img_copy = image.copy()
        image_expanded = np.expand_dims(image, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        min_score_thresh = 0.60
        if len(boxes)>0:
            all_boxes, classes = boxes[0][scores[0]>=min_score_thresh], classes[0][scores[0]>=min_score_thresh]
            # Draw the results of the detection (aka 'visulaize the results')
            # boxed_image, all_boxes = vis_util.visualize_boxes_and_labels_on_image_array(
            #     image,
            #     np.squeeze(boxes),
            #     np.squeeze(classes).astype(np.int32),
            #     np.squeeze(scores),
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=3,
            #     min_score_thresh=0.60)
            #
            for i, box in enumerate(all_boxes):
                ymin, xmin, ymax, xmax = box
                shape = np.shape(image)
                im_width, im_height = shape[1], shape[0]
                (left, right, top, bottom) = (int(xmin * im_width),int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
                cropped = img_copy[top:bottom,left:right]
                cv2.rectangle(image,(int(left),int(top)),(int(right),int(bottom)), (0,255,0), 2)
                cv2.putText(image, "identity_card", (left,bottom), cv2.FONT_HERSHEY_COMPLEX_SMALL,2, [0,255,0], 2)
                # Using Image to crop and save the extracted copied image
                cropped_image.append(cropped)
        else:
            print("NO Document detected")

        boxed_image, cropped_image, all_boxes = image, cropped_image, all_boxes

    except Exception as e:
        print("Failed to crop, error: ".format(str(e)))

    return boxed_image, cropped_image , all_boxes






def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def rotate_and_adjust(cropped_images, original_image):
    img_copy = original_image.copy()
    adjusted_images = []
    for i, im_crp in enumerate(cropped_images):
        cr_height, cr_width,_ = im_crp.shape
        cr_area = cr_height*cr_width
        im_height , im_width,_ = original_image.shape
        im_area = im_height*im_width
        crp_area_prcent = cr_area*100/im_area
        if crp_area_prcent>90:
            im_crp = img_copy
        if cr_height>cr_width:
            yen_threshold = threshold_yen(im_crp)
            bright = rescale_intensity(im_crp, (0, yen_threshold), (0, 255))
            img_morphed = preprocess_image(bright, process='cubic')
            try:
                angle = 360 - int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(img_morphed)).group(0))
                im_crp = rotate_image(im_crp, angle)
            except:
                pass

        adjusted_images.append(im_crp)
    return adjusted_images



