import torch

# Load the pre-trained YOLOv5 model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='.venv\\objectDetect\\yolov5\\best.pt')
model.conf = 0.1  # confidence threshold (0-1)
# For a given image
img = 'C:\\Users\\02009964\\OneDrive - pccw.com\\Desktop\\mic\\side_project\\Au\\pre_course\\.venv\\objectDetect\\yolov5\\tester.jpg'  # Can be a file, PIL image, or URL

# Inference
results = model(img)

# Results
results.print()  # Print results to console
results.show()  # Show the detected objects in the image
