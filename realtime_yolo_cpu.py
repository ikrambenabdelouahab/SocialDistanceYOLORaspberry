import cv2
import numpy as np
import time
from scipy.spatial import distance
import csv
from datetime import datetime

# Load Yolo
net = cv2.dnn.readNet("src/yolov3-tiny.weights", "src/yolov3-tiny.cfg")
classes = []
with open("src/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Loading camera
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
cnt = 0

row_list = [["detection_id","p1" , "p2", "distance","msg", "DateTime"]]
detection_id = 1

while True:
    _, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    midpoints = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                cnt = cnt + 1

                midp = (center_x, center_y)
                midpoints.append([midp, cnt])
                num = len(midpoints)
                print(num)
                # Compute distance between objects
                for m in range(num):
                    for n in range(m+1,num):
                        if m!=n:
                            # GET datetime NOW
                            now = datetime.now()
                            current_time = now.strftime("%H:%M:%S")

                            dst = distance.euclidean(midpoints[m][0], midpoints[n][0])
                            p1 = midpoints[m][1]
                            p2 = midpoints[n][1]
                            print("Distance entre personne ", p1, " et personne", p2, " ====== ", dst)
                            if (dst <=200):
                                warning_msg = "ALERT"
                                print("ALERT")
                            else:
                                warning_msg = "normal"
                                print("Normal")
                            detection_id = detection_id + 1
                            # LOG file
                            #save to csv file
                            with open('log.csv', 'w') as file:
                                writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
                                row_list.append([detection_id, p1, p2, dst, warning_msg, current_time])
                                detection_id = detection_id + 1
                                writer.writerows(row_list)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()