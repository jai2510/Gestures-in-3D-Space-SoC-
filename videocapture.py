import numpy as np
import cv2
import pyautogui

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    roi = cv2.rectangle(frame, (200,200), (500,600), (0,0,255), 3)
    cv2.imshow('frame',frame)
    cv2.imwrite("/Users/jai/hand_pointer/test/blah/my_gesture.png",
                frame[200:600,200:500])
    
    predictions=model.predict_generator(test_generator)
    print(np.argmax(predictions))
    print(train_generator.class_indices)
    
    if np.argmax(predictions) == 0:
        count += 1
    if count > 0:
        if np.argmax(predictions) == 0                                                                 :
            pyautogui.press("space")
    
                                                                                            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
                     