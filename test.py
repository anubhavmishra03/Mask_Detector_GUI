import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained MobileNetV2 model
model = load_model("mask_detector_mobilenet.h5")

classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

labels_dict={1:"without_mask",0:"with_mask"}
color_dict={1:(0,0,255),0:(0,255,0)}

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or specify the path to a video file

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = classifier.detectMultiScale(frame, 1.3, 5)

    for x,y,w,h in faces:
        face_img = frame[y:y+h, x:x+w]
        img = cv2.resize(face_img, (224, 224))
        data = []
        data.append(img)
        data = np.array(data)
        data = data/255
        predictions = model.predict(data)
        if predictions<=0.5:
            label = 0
        else:
            label = 1

        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
    # Preprocess input frame
    # Resize frame to match input size of MobileNetV2
    #input_data = np.expand_dims(frame, axis=0)  # Add batch dimension

    #input_data = preprocess_input(input_data)  # Preprocess input (normalize pixel values)
    
    # Perform inference

    #predicted_labels = decode_predictions(predictions, top=5)[0]  # Decode predictions

    #print(predicted_labels)
    
    # Display predicted labels on frame
    #label_str = ''
    #for i, (imagenet_id, label, score) in enumerate(predicted_labels):
    #    label_str += f"{label}: {score:.2f}\n"
    
    #cv2.putText(frame, predicted_labels, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
