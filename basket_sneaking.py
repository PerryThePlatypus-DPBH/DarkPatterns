import re
from utils.another_helper import get_only_text_bboxes
from ultralytics import YOLO
import pytesseract
import cv2
import languagemodels as lm

def crop_image(image, coords):
    x1, y1, x2, y2 = coords
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    return cropped_image

def highest_price(image):
    model = YOLO("runs/detect/train/weights/best.pt")

    bboxes = get_only_text_bboxes(model, image)

    bb = []
    for i in bboxes:
        bb.append(i.tolist()[0])

    sorted_bb = sorted(bb, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))[::-1]

    img = cv2.imread(image)
    im_list= []

    for i in sorted_bb:
        im_list.append(crop_image(img, i))

    text_list = []
    for image in im_list:
        text = pytesseract.image_to_string(image)
        text_list.append(text)
    text_list = [p for p in text_list if any(char.isdigit() for char in p)]

    price = []
    for text in text_list:
        result = lm.do("What is the price in the following line, note sometimes it's not formatted and maybe werid, pls return empty if ther's no price seen" + text)
        price.append(result)

    price = [p for p in price if any(char.isdigit() for char in p)]

    digits = 0
    for p in price:
        match = re.search(r'\d+', p)
        if match:
            digits = match.group()
            digits = int(digits)
            print(digits)

            with open("high.txt", "r") as file:
                existing_digit = int(file.read())
            
            if digits > existing_digit:
                with open("high.txt", "w") as file:
                    file.write(str(digits))
                return True
            else:
                return False
    
    
print("res: ", highest_price("test/test.jpg"))