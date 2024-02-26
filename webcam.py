import cv2
import torch

# Load the pre-trained YOLOv5 model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='.venv\\objectDetect\\yolov5\\best.pt')
#model = torch.hub.load('/Users/macmini04/Documents/yolov5','custom', path='best.pt',force_reload=True,source='local', pretrained =Flase)
model.conf = 0.01  # confidence threshold (0-1)

def detect_objects(model, frame):
    # Convert the color from BGR (OpenCV) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform inference
    results = model(frame_rgb)
    
    # Extract data
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord

def display_detected_objects(frame, labels, cords):
    for label, cord in zip(labels, cords):
        x1, y1, x2, y2, conf = cord
        start_point = int(x1 * frame.shape[1]), int(y1 * frame.shape[0])
        end_point = int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
        cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.putText(frame, f'{model.names[int(label)]} {conf:.2f}', start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for web camera

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        labels, cords = detect_objects(model, frame)
        display_detected_objects(frame, labels, cords)

        cv2.imshow('YOLOv5 Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
