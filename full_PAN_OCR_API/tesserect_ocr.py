import pytesseract
#pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
import cv2
import os
import pickle
import json
import re

from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave




def preprocess_image(image, process = "thresh"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if process == "thresh":
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    elif process == "adaptive":
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    elif process == "linear":
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    elif process == "cubic":
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    elif process == "blur":
        gray = cv2.medianBlur(gray, 3)

    elif process == "bilateral":
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

    elif process == "gauss":
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    else:
        print("Process method not available. Choose another option.")

    return  gray
#
# pan_data_path = "out/test4"
# with open(pan_data_path+ "/pan_data.pkl","rb") as f:
#     pan_data = pickle.load(f)




def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf




def get_text(image):
    yen_threshold = threshold_yen(image)
    bright = rescale_intensity(image, (0, yen_threshold), (0, 255))
    img_morphed = preprocess_image(bright, process='cubic')
#    text = pytesseract.image_to_string(img_morphed, lang="eng", config="--oem 3")
    df = pytesseract.image_to_data(img_morphed, lang="eng", config="--oem 3", output_type = "data.frame")
    ocr_conf = round(df.loc[~df.text.isnull(), ["conf"]].mean().fillna(0)[0], 2)
    text = " ".join(df.loc[~df.text.isnull(), "text"].tolist())
    text1 = text.strip()
    text2 = re.sub(r'[^ \nA-Za-z0-9/]+', '', text1)
    text2 = text2.replace("\n", " ")
    text2 = text2.strip()
    return text2, ocr_conf



################ OLD GET TEXT FUNCTION ##############

#def get_text(image):
#    img_denoised = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
#    img_morphed = preprocess_image(img_denoised, process= 'linear')
#    text = pytesseract.image_to_string(img_morphed, lang='eng')
#    text1 = text.strip()
#    text2 = re.sub(r'[^ \nA-Za-z0-9/]+', '', text1)
#    text2 = text2.replace("\n", " ")
#    return text2, img_denoised, img_morphed

# cv2.imshow("dob_denoised", pan_img_denoised)
# cv2.imshow("dob_morph", pan_img_morphed)
# print(result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

########### VERY IMPORTANT STUFF #################
#
#from skimage.filters import threshold_yen
#from skimage.exposure import rescale_intensity
#from skimage.io import imread, imsave
#import cv2
#
#
#def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
#    if brightness != 0:
#        if brightness > 0:
#            shadow = brightness
#            highlight = 255
#        else:
#            shadow = 0
#            highlight = 255 + brightness
#        alpha_b = (highlight - shadow)/255
#        gamma_b = shadow
#        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
#    else:
#        buf = input_img.copy()
#
#    if contrast != 0:
#        f = 131*(contrast + 127)/(127*(131-contrast))
#        alpha_c = f
#        gamma_c = 127*(1-f)
#
#        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
#
#    return buf
#
#

#img = cv2.imread(r"out/336961230/name.jpg")
#yen_threshold = threshold_yen(img)
#bright = rescale_intensity(img, (0, yen_threshold), (0, 255))
#img_morphed = preprocess_image(bright, process='cubic')
#ocr = pytesseract.image_to_string(img_morphed, lang="eng", config="--oem 3")
#print(ocr)

#cv2.imshow("bright", bright)
#cv2.imshow("morph_bright", img_morphed)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#
#
#out_img = apply_brightness_contrast(img, 64, 64)
#img_morphed1 = preprocess_image(out_img, process='cubic')
#ocr = pytesseract.image_to_string(img_morphed1, lang="eng", config="--oem 3")
#print(ocr)
#
#cv2.imshow("org", out_img)
#cv2.imshow("img_morphed", img_morphed1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
