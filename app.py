# import libraries
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


# def hand detection class
class DrawingApp():
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5 , model_path = 'cnn-mnist-model.h5'):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        # initialize model
        self.model = tf.keras.models.load_model(model_path)
        self.points = []
        
        # initialize mediapipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=self.detection_confidence,min_tracking_confidence=self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        
        # initialize webcam
        self.cap = cv2.VideoCapture(0)
    
    # get the coordinates of index finger and thumb
    def get_index_thumb_coordinates(self, img, results):
        # get the landmark of index finger and thumb
        index_finger = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        
        # get the coordinates of index finger and thumb
        index_finger_x = int(index_finger.x * img.shape[1])
        index_finger_y = int(index_finger.y * img.shape[0])
        thumb_x = int(thumb.x * img.shape[1])
        thumb_y = int(thumb.y * img.shape[0])
        
        return [index_finger_x, index_finger_y, thumb_x, thumb_y]
    # return bool if index finger and thumb distance is less than 50
    def is_index_thumb_close(self, img, results):
        # get the coordinates of index finger and thumb
        index_finger_x, index_finger_y, thumb_x, thumb_y = self.get_index_thumb_coordinates(img, results)
        
        # calculate the distance between index finger and thumb
        distance = np.sqrt(np.square(index_finger_x - thumb_x) + np.square(index_finger_y - thumb_y))
        
        # return bool if distance is less than 50
        return distance < 35
    
    # return bool if fist is detected
    def is_fist(self, img, results):
        index_finger = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        index_finger_MCP = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_finger_MCP = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_finger_MCP = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_MCP = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        
        # check if index finger, middle finger, ring finger and pinky are closed
        if index_finger.y > index_finger_MCP.y and middle_finger.y > middle_finger_MCP.y and ring_finger.y > ring_finger_MCP.y and pinky.y > pinky_MCP.y:
            return True
   
    # draw a contour on the image
    def draw_contour(self, img, img2):
        # draw a line between the points
        for i in range(1, len(self.points)):
            cv2.line(img, self.points[i - 1], self.points[i], (255, 0, 255), 30)
            cv2.line(img2, self.points[i - 1], self.points[i], (255, 0, 255), 30)
    
    # predict the digit
    def predict_digit(self, img1):
        # resize the image
        img = cv2.resize(img1, (28, 28))

        # normalize the image
        img = img / 255.0

        # reshape the image
        img = img.reshape(1, 28, 28, 1)

        # predict the digit
        return np.argmax(self.model.predict(img))
    
    # open the camera and detect the hand
    def start(self):

        # blank window settings
        red_low = np.array([0, 0, 255])
        red_high = np.array([255, 0, 255])  
        
        # loop until the camera is open
        while self.cap.isOpened():
            # read the camera
            success, img = self.cap.read()
            if not success:
                # if not success, break the loop
                print("Ignoring empty camera frame.")
                continue
            
            red_mask = cv2.inRange(img, red_low, red_high)
            # convert the image to RGB flip it horizontally
            img = cv2.flip(img, 1)
            
            # process the image
            results = self.hands.process(img)
            
            self.draw_contour(img,red_mask)
            
            # write prediction on the image
            predict = self.predict_digit(red_mask)
            cv2.putText(img, str(predict), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            # draw a circle inbetween the index finger and thumb
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # get the coordinates of index finger and thumb
                    index_finger_x, index_finger_y, thumb_x, thumb_y = self.get_index_thumb_coordinates(img, results)
                    
                    # draw a circle inbetween the index finger and thumb
                    cv2.circle(img, ((index_finger_x + thumb_x) // 2, (index_finger_y + thumb_y) // 2), 10, (255, 0, 255), cv2.FILLED)
                    
                    # check if index finger and thumb is close
                    if self.is_index_thumb_close(img, results):
                        # draw a circle in the middle of the line
                        cv2.circle(img, ((index_finger_x + thumb_x) // 2, (index_finger_y + thumb_y) // 2), 10, (0, 255, 0), cv2.FILLED)
                        # add the point to the list
                        self.points.append(((index_finger_x + thumb_x) // 2, (index_finger_y + thumb_y) // 2))
                    elif self.is_fist( img, results):
                        self.points = []
                
            cv2.imshow("window_masked", red_mask)    
            # show the image
            cv2.imshow("Image", img)
            
            # break the loop if close is pressed or esc is pressed
            if cv2.waitKey(2) & 0xFF == 27:
                break
        
        # release the camera and destroy all windows
        self.cap.release()
        cv2.destroyAllWindows()
        # import matplotlib.pyplot as plt
        # return red_mask
    
#main function
if __name__ == "__main__":    
    # initialize the hand detection class
    app = DrawingApp()
    app.start()