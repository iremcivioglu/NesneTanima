import cv2
import numpy as np
import argparse
import os

def yolo():

    net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3.cfg')

    image = input("Görüntü : ")
    img = cv2.imread(image)
    # img = cv2.resize(img, (400, 400))
    t = True

    class_names = open("classes.txt").read().strip().split("\n")


    while t:
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        layers_names = net.getUnconnectedOutLayersNames()
        outputs = net.forward(layers_names)

        bounding_box = []
        confidences = []
        classIDs = []

        for outs in outputs:
            for detection in outs:  #
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    bounding_box.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    classIDs.append(class_id)

        index = cv2.dnn.NMSBoxes(bounding_box, confidences, 0.5, 0.4) 

        for i in range(len(bounding_box)):
            if i in index:
                x, y, w, h = bounding_box[i]
                id = int(classIDs[i])
                label = class_names[id]
                confidence = str(round(confidences[i],2))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(img, label + " " + confidence, (x, y + int(1.15*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            t = False

            cv2.imshow("Goruntu", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('detect', help='Görüntüdeki nesneleri tespit etmek için komutu giriniz.')
    parser.set_defaults(func=yolo)
    args = parser.parse_args()
    args.func()


if __name__ == '__main__':
    main()
