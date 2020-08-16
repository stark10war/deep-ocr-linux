import argparse
import cv2
import numpy as np
import os, io, sys
from PIL import Image
import base64
from pdf2image import convert_from_path
from flask import jsonify, render_template
from flask_cors import CORS
import pytesseract
from tesserect_ocr import preprocess_image, get_text
#os.chdir(r"C:\Users\shash\OneDrive\Documents\Shashank\YOLO OCR\full_PAN_OCR_API")
from flask import Flask, flash, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from id_card_detector import id_card_detection_image
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity

#pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

class ocr_yolo_models:
    # initializing with paths and loading the model
    def __init__(self, config, weights, names,CONF_THRESH = 0.2, NMS_THRESH= 0.2):
        self.config = config
        self.weights = weights
        self.names = names
        self.CONF_THRESH, self.NMS_THRESH = CONF_THRESH, NMS_THRESH
        # Load the network
        self.net = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Draw the filtered bounding boxes with their class to the image
        with open(self.names, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    # running forward pass to get all b_box predictions with classes
    def get_detections(self,image):
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
        # Get the output layer from YOLO
        height, width = image.shape[:2]
        layers = self.net.getLayerNames()
        output_layers = [layers[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.net.setInput(blob)
        layer_outputs = self.net.forward(output_layers)
        class_ids, confidences, b_boxes, indices = [], [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.CONF_THRESH:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    b_boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))
        if len(b_boxes)>0:
            indices = cv2.dnn.NMSBoxes(b_boxes, confidences, self.CONF_THRESH, self.NMS_THRESH).flatten().tolist()

        return b_boxes, confidences, class_ids, indices

    # drawing bounding boxes
    def draw_boxes(self, image, b_boxes, confidences, class_ids, indices):
        # set bounding box colours
        # colors = np.random.uniform(0, 255, size=(len(classes), 3))
        name_col = [0, 0, 255]
        fname_col = [0, 255, 255]
        dob_col = [0, 255, 128]
        pan_col = [255, 0, 0]
        colors = [name_col, fname_col, dob_col, pan_col]
        for index in indices:
            x, y, w, h = [i if i >= 0 else 0 for i in b_boxes[index]]
            class_name = self.classes[class_ids[index]]
            confidence_val = round(confidences[index], 2)
            try:
                cv2.rectangle(image, (x, y), (x + w, y + h), colors[index], 2)
                cv2.putText(image, class_name + " : {}%".format(confidence_val), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,colors[index], 2)
            except:
                cv2.rectangle(image, (x, y), (x + w, y + h), [255,0,0], 2)
                cv2.putText(image, class_name + " : {}%".format(confidence_val), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, [255,0,0], 2)

        return image

    # Cropping entites to feed into OCR , returns cordinates , confidence and cropped images
    def crop_entities(self,image,b_boxes, confidences, class_ids, indices):
        # crop objects
        entities = {}
        for index in indices:
            x, y, w, h = [i if i>=0 else 0 for i in b_boxes[index]]
            class_name = self.classes[class_ids[index]]
            confidence_val = confidences[index]
            crop_img = image[y:y + h, x:x + w]
            entities[class_name] = {"cords":(x,y,w,h),"conf": confidence_val,"img": crop_img}

        return entities


# Running OCR on Cropped entites with image processing. Generates final OCR results with confidence and clean text
def run_ocr_pan(cropped_entites):
    result = {"name": "", "father_name": "", "dob":"", "pan": ""}
    for key in cropped_entites:
        if len(cropped_entites[key]) > 1:
            print(key)
            text, ocr_conf = get_text(cropped_entites[key]["img"])
            entity_conf = round(cropped_entites[key]["conf"]*100,2)
            result[key] = {"text": text, "entity_conf": entity_conf, "ocr_conf": ocr_conf}
    return result

def run_raw_ocr(image):
    yen_threshold = threshold_yen(image)
    bright = rescale_intensity(image, (0, yen_threshold), (0, 255))
    img_morphed = preprocess_image(bright, process='cubic')
    text = pytesseract.image_to_string(img_morphed, lang="eng", config="--oem 3")
    return text


# conver URI / base64 encoded data to Image .
def get_image_from_uri(data_uri):
    encoded_data = data_uri.split(b',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# Convert each page of PDF to image
def pdf_to_images(pdf_path):
    images = []
    images1 = convert_from_path(pdf_path)
    for image in images1:
        img = np.array(image)
        images.append(img)
    return images


# convert image to bytes to send response image through API.
def image_to_byte(image):
    img_out = Image.fromarray(image.astype("uint8"))
    rawBytes = io.BytesIO()
    img_out.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return str(img_base64)


def get_encoded_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image.astype("uint8"))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return my_encoded_img



# Configuring files and model locations.
#POPPLER_PATH =r"C:\Users\shash\OneDrive\Documents\Shashank\YOLO OCR\full_PAN_OCR_API\poppler-0.68.0\bin"

######## model and config files locations #############
# doc classification
CONFIG_DOC_CLF  = "doc_classifier/yolov3-tiny_custom.cfg"
WEIGHTS_DOC_CLF = "doc_classifier/backup/yolov3-tiny_custom_last.weights"
NAMES_DOC_CLF = "doc_classifier/custom.names"


# PAN Format classification
CONFIG_PAN_CLF = "pan_format_classifier/yolov3-tiny_custom.cfg"
WEIGHTS_PAN_CLF = "pan_format_classifier/backup/yolov3-tiny_custom_last.weights"
NAMES_PAN_CLF = "pan_format_classifier/custom.names"

# PAN format1 YOLO model for entity detection
CONFIG_PAN_F1_CLF = "pan_format1/yolov3-tiny_custom.cfg"
WEIGHTS_PAN_F1_CLF = "pan_format1/backup/yolov3-tiny_custom_last_5000.weights"
NAMES_PAN_F1_CLF = "pan_format1/custom.names"

# PAN format2 YOLO model for entity detection
CONFIG_PAN_F2_CLF  = "pan_format2/yolov3-tiny_custom.cfg"
WEIGHTS_PAN_F2_CLF = "pan_format2/backup/yolov3-tiny_custom_last.weights"
NAMES_PAN_F2_CLF = "pan_format2/custom.names"


#### initializing all models
doc_clf_model = ocr_yolo_models(config=CONFIG_DOC_CLF,
                                weights=WEIGHTS_DOC_CLF,
                                names=NAMES_DOC_CLF)

pan_clf_model = ocr_yolo_models(config=CONFIG_PAN_CLF,
                                weights=WEIGHTS_PAN_CLF,
                                names=NAMES_PAN_CLF,
                                CONF_THRESH=0.5,
                                NMS_THRESH=0.5)

pan_f1_model = ocr_yolo_models(config=CONFIG_PAN_F1_CLF,
                               weights=WEIGHTS_PAN_F1_CLF,
                               names=NAMES_PAN_F1_CLF)

pan_f2_model = ocr_yolo_models(config=CONFIG_PAN_F2_CLF,
                               weights=WEIGHTS_PAN_F2_CLF,
                               names=NAMES_PAN_F2_CLF)





app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'input_files'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


#initialting cropping modle run for boost
boost = np.zeros((500,500,3))
id_card_detection_image.crop_document(boost)


@app.route('/', methods=['GET'])
def upload_f():
   return render_template('upload2.html')

@app.route('/image', methods=['POST'])
def classify_document():
    f = request.files['file']
    file_name = secure_filename(f.filename)
    file_ext = file_name.split(".")[-1]
    print(file_name, file_ext)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
    if file_ext in ["pdf", "PDF"]:
        images = pdf_to_images(pdf_path="input_files/{}".format(file_name))
        images = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in images]
    else:
        images = [cv2.imread("input_files/{}".format(file_name))]

    # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
    pages = {"meta":{"id":"","doc_extension":"", "page_count": "","doc_count":"", "status":""}, "data": {}, "out_images": {"img": "", "img1": "", "img2": ""}}
    pages["meta"]["id"] = file_name
    pages["meta"]["doc_extension"] = file_ext
    pages["meta"]["page_count"] = len(images)
    doc_counter = 0
    for i, img in enumerate(images):
        page_no = "page_{}".format(str(i+1))
        img_copy1 = img.copy()
        ################# RUNNING DOCUMENT CROPPING #############################
        docs = {}
        boxed_image, cropped_images, all_boxes = id_card_detection_image.crop_document(img.copy())
        if len(all_boxes)>0:
            doc_counter += 1
            adjusted_images = id_card_detection_image.rotate_and_adjust(cropped_images, img)
        else:
            adjusted_images = [img]
        #writing image
        out_file  = file_name.split(".")[0]
#        cv2.imwrite("out/{}_{}_boxed.jpg".format(out_file, page_no), boxed_image)
        img_byte_boxed = get_encoded_img(boxed_image)
        pages["out_images"]["img"] = img_byte_boxed
        for i_crp, im_croped in enumerate(adjusted_images):
            doc_result = {"doc_type": "", "doc_sub_class": "", "message":"", "ocr_results": {}}
            doc_no = "doc_{}".format(str(i_crp+1))
            ########### RUNNING DOCUMENT CLASSIFICATION ON EACH CROPPED IMAGE ####################
            b_boxes, confidences, class_ids, indices = doc_clf_model.get_detections(im_croped)
            drawn_img = []
            if indices:
                drawn_img = doc_clf_model.draw_boxes(im_croped.copy(),b_boxes, confidences, class_ids, indices)
                max_index = confidences.index(max(confidences))
                document_class = doc_clf_model.classes[class_ids[max_index]]
                doc_clf_conf = round(confidences[max_index], 2)
            else:
                document_class = "others"
                doc_clf_conf = 1
            print(document_class)

            doc_result["doc_type"] = document_class

            ######### PAN FORMAT CLASSIFICATION ########################
            if document_class =="pan_landmark":
                b_boxes, confidences, class_ids, indices = pan_clf_model.get_detections(im_croped)
                cv2.putText(drawn_img, "PAN CARD", (40,drawn_img.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255), 3)
                if indices:
                    drawn_img = pan_clf_model.draw_boxes(drawn_img, b_boxes, confidences, class_ids, indices)
                    max_index = confidences.index(max(confidences))
                    pan_type_class = doc_clf_model.classes[class_ids[max_index]]
                    final_confidence = round(confidences[max_index], 2)
                    pan_format = "f2"
                else:
                    pan_format = "f1"

                doc_result["doc_sub_class"] = pan_format
                ########## Running Entity detection Model #######################
                entity_model = eval("pan_{}_model".format(pan_format))
                b_boxes, confidences, class_ids, indices = entity_model.get_detections(im_croped)
                if indices:
                    drawn_img_entity = entity_model.draw_boxes(im_croped.copy(), b_boxes, confidences, class_ids, indices)
                    entities = entity_model.crop_entities(im_croped.copy(), b_boxes=b_boxes,
                                                           confidences=confidences,
                                                           class_ids=class_ids,
                                                           indices=indices)
                    img_byte_entity = get_encoded_img(drawn_img_entity)
                    pages["out_images"]["img2"] = img_byte_entity
                    ####### Running OCR engine ###############
                    results = run_ocr_pan(entities)
                    doc_result["message"] = "PASS"
                    doc_result["ocr_results"] = results
                else:
                    #No entites detected
                    doc_result["message"] = "No Entites found"
                    print("no entities detected")
            elif document_class == "aadhar_landmark":
                cv2.putText(drawn_img, "AADHAAR CARD", (40,drawn_img.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255), 3)
                doc_result["message"] = "No OCR model for Aadhaar"
                doc_result["ocr_results"] = run_raw_ocr(im_croped)
            else:
                doc_result["ocr_results"] = ""
                doc_result["message"] = "document is not PAN/Aadhaar"
                print("Document is unrecognized")

            if len(drawn_img)>0:
#                cv2.imwrite("out/{}_{}_drawn.jpg".format(out_file, i_crp), drawn_img)
                img_byte_clf = get_encoded_img(drawn_img)
                pages["out_images"]["img1"] = img_byte_clf


            docs[doc_no] = doc_result

        pages["data"][page_no]= docs
    pages["meta"]["doc_count"] = doc_counter

    return jsonify(pages)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)




