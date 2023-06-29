import cv2

# Load the pre-trained cascade classifier for license plate detection
plate_cascade = cv2.CascadeClassifier('path/to/haarcascade_license_plate.xml')

# Open video capture from a video file or camera
video_capture = cv2.VideoCapture('path/to/video.mp4')  # Replace with 0 for live camera feed

while video_capture.isOpened():
    # Read a frame from the video stream
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect license plates in the frame
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Iterate over detected license plates and draw rectangles around them
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract the region of interest (ROI) containing the license plate
        plate_roi = gray[y:y + h, x:x + w]
        
        # Perform further processing on the ROI, such as character recognition
        # You can use OCR libraries or custom algorithms to recognize characters
    
    # Display the frame with license plate detection
    cv2.imshow('License Plate Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
