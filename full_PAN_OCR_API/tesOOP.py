import argparse

import cv2
import numpy as np
import os
import pickle
import json
import pandas as pd
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
from tesserect_ocr import preprocess_image, get_text
os.chdir(r"C:\Users\shash\OneDrive\Documents\Shashank\YOLO OCR\full_PAN_OCR_API")


class ocr_yolo_models:

    def __init__(self, config, weights, names,CONF_THRESH, NMS_THRESH):
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




    def get_detections(self,image):
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
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
            cv2.rectangle(image, (x, y), (x + w, y + h), [255, 0, 0], 2)
            try:
                cv2.putText(image, class_name + " : {}%".format(confidence_val), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,colors[index], 2)
            except:
                cv2.putText(image, class_name + " : {}%".format(confidence_val), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, [255,0,0], 2)

        return image

    def crop_entities(self,image,b_boxes, confidences, class_ids, indices):
        # crop objects
        entities = {}
        for index in indices:
            x, y, w, h = [i if i>=0 else 0 for i in b_boxes[index]]
            class_name = self.classes[class_ids[index]]
            confidence_val = confidences[index]
            crop_img = img[y:y + h, x:x + w]
            entities[class_name] = {"cords":(x,y,w,h),"conf": confidence_val,"img": crop_img}

        return entities



def run_ocr(cropped_entites):
    result = {"id": "" ,"name": "", "father_name": "", "dob":"", "pan": ""}
    for key in cropped_entites:
        if len(cropped_entites[key]) > 1:
            print(key)
            text, df, img_morphed = get_text(cropped_entites[key]["img"])
            ocr_conf = round(df.loc[~df.text.isnull(), ["conf"]].mean().fillna(0)[0],2)
            entity_conf = round(cropped_entites[key]["conf"]*100,2)
            result[key] = {"text": text, "entity_conf":entity_conf, "ocr_conf": ocr_conf}
    return result




format1_model = ocr_yolo_models(config="pan_format1/yolov3-tiny_custom.cfg",
                                weights="pan_format1/backup/yolov3-tiny_custom_last_5000.weights",
                                names="pan_format1/custom.names",
                                CONF_THRESH=0.2,
                                NMS_THRESH=0.2)

img = cv2.imread(r"f28.JPG")

b_boxes, confidences, class_ids, indices = format1_model.get_detections(img)

new_img = format1_model.draw_boxes(image=img.copy(),
                                   b_boxes=b_boxes,
                                   confidences=confidences,
                                   class_ids=class_ids,
                                   indices=indices)






config  = "pan_format2/yolov3-tiny_custom.cfg"
weights = "pan_format2/backup/yolov3-tiny_custom_last.weights"
names = "pan_format2/custom.names"



format2_model = ocr_yolo_models(config=config,
                                weights=weights,
                                names=names,
                                CONF_THRESH=0.2,
                                NMS_THRESH=0.2)

b_boxes, confidences, class_ids, indices = format2_model.get_detections(img)

new_img = format2_model.draw_boxes(image=img.copy(),
                                   b_boxes=b_boxes,
                                   confidences=confidences,
                                   class_ids=class_ids,
                                   indices=indices)


entities = format2_model.crop_entities(img.copy(), b_boxes=b_boxes,
                                   confidences=confidences,
                                   class_ids=class_ids,
                                   indices=indices)


entities.keys()

result= run_ocr(cropped_entites=entities)
result.keys()

cv2.imshow("o",entities["pan"]["img"])
cv2.imshow("b",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

doc_clf_model = ocr_yolo_models(config  = "doc_classifier/yolov3-tiny_custom.cfg",
                                weights = "doc_classifier/backup/yolov3-tiny_custom_last.weights",
                                names = "doc_classifier/custom.names",
                                CONF_THRESH=0.2,
                                NMS_THRESH=0.2)



b_boxes, confidences, class_ids, indices = doc_clf_model.get_detections(img)

new_img =  doc_clf_model.draw_boxes(image=img.copy(),
                                   b_boxes=b_boxes,
                                   confidences=confidences,
                                   class_ids=class_ids,
                                   indices=indices)

entities = doc_clf_model.crop_entities(img, b_boxes=b_boxes,
                                   confidences=confidences,
                                   class_ids=class_ids,
                                   indices=indices)

entities.keys()

crp = entities["pan_landmark"]["img"]
cv2.imshow("o",crp)
cv2.imshow("b",new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


doc_clf_model = ocr_yolo_models(config  = "doc_classifier/yolov3-tiny_custom.cfg",
                                weights = "doc_classifier/backup/yolov3-tiny_custom_last.weights",
                                names = "doc_classifier/custom.names",
                                CONF_THRESH=0.2,
                                NMS_THRESH=0.2)



b_boxes, confidences, class_ids, indices = doc_clf_model.get_detections(img)

new_img =  doc_clf_model.draw_boxes(image=img.copy(),
                                   b_boxes=b_boxes,
                                   confidences=confidences,
                                   class_ids=class_ids,
                                   indices=indices)



class_ids
cv2.imshow("o",img)
cv2.imshow("b",new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
master_db = pd.DataFrame()

for i , image in enumerate(os.listdir(INPUT_FOLDER)):
    img_path = os.path.join(INPUT_FOLDER,image)
    img_folder = image[:len(image)-4]
    out_path = "out/{}".format(img_folder)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    try:
        print(img_path)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)
        class_ids, confidences, b_boxes = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONF_THRESH:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    b_boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))


        # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()



        # crop objects
        pan_obj = {"name":"", "father_name": "", "dob":"", "pan":""}
        for index in indices:
            x, y, w, h = [i if i>=0 else 0 for i in b_boxes[index]]
            class_name = classes[class_ids[index]]
            confidence_val = confidences[index]
            crop_img = img[y:y + h, x:x + w]
            cv2.imwrite(out_path+"/{}.jpg".format(class_name), crop_img)
            pan_obj[class_name] = [(x,y,w,h), confidence_val, crop_img]

        with open(out_path+"/pan_data.pkl", 'wb') as f:
            pickle.dump(pan_obj, f)

        with open(out_path+ "/pan_data.pkl","rb") as f:
            pan_obj = pickle.load(f)


        #creating bounding boxes
        for index in indices:
            x, y, w, h = b_boxes[index]
            class_name = classes[class_ids[index]]
            confidence_val = confidences[index]
            cv2.rectangle(img, (x, y), (x + w, y + h), colors[index], 2)
            cv2.putText(img, class_name, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[index], 2)

        cv2.imwrite(out_path + "/predicted.jpg", img)
        cv2.imwrite("predicted/predicted_{}.jpg".format(img_folder), img)

        # extracting text using Tesseract OCR
        result = {"id": "" ,"name": "", "father_name": "", "dob":"", "pan": "", "out_path" : ""}
        result["id"] = img_folder
        result["out_path"] = out_path
        for key in pan_obj:
            if len(pan_obj[key]) > 1:
                text, img_denoised, img_morphed = get_text(pan_obj[key][2])
                prob_score = round(pan_obj[key][1]*100,2)
                result[key] = text + " ; {}%".format(prob_score)

        # print('''Name: {}\nFather's Name: {}\nDOB: {}\nPAN: {}'''.format(name, father_name, dob, pan))
        with open(out_path + "/out_result.json", 'w') as f:
            json.dump(result, f, indent=4)

        df = pd.DataFrame.from_dict([result], orient='columns')
        master_db = master_db.append(df)

    except:
        print("Failed for {}".format(img_folder))

master_db.to_csv("Master_db.csv", index = False)


# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

