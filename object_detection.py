import cv2

prototxt = './MobileNetSSD_deploy.prototxt.txt'

model = './MobileNetSSD_deploy.caffemodel'

classes = {0:"carton", 
        1:"organicos", 
        2:"plastico", 
        3:"toxicos", 
        4:"vidrio"}


net = cv2.dnn.readNetFromCaffe(prototxt, model)

cap = cv2.VideoCapture(0)
while True:
     ret, frame, = cap.read()
     if ret == False:
          break
     height, width, _ = frame.shape
     frame_resized = cv2.resize(frame, (300, 300))
     # Create a blob
     blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))
     #print("blob.shape:", blob.shape)
     # ----------- DETECTIONS AND PREDICTIONS -----------
     net.setInput(blob)
     detections = net.forward()
     for detection in detections[0][0]:
          #print(detection)
          if detection[0] > 0.45:
               label = classes[detection[1]]
               #print("Label:", label)
               box = detection[3:4] * [width, height, width, height]
               x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
               cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
               cv2.putText(frame, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 0, 0), 2)
               cv2.putText(frame, label, (x_start, y_start - 25), 1, 1.5, (0, 255, 255), 2)
     cv2.imshow("Frame", frame)
     if cv2.waitKey(1) & 0xFF == 27:
          break
cap.release()
cv2.destroyAllWindows()