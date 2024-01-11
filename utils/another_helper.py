from ultralytics import YOLO

def get_only_text_bboxes(model: YOLO, img_path):
    results = model(img_path)

    filtered_boxes = []
    for result in results:
        for box in result.boxes:
            if box.cls == 6:
                filtered_boxes.append(box.xyxy)

    return filtered_boxes