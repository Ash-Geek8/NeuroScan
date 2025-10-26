from flask import Flask, render_template, request, redirect, url_for, g
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER']="static\\images"

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Here you would handle file upload
        upload_image=request.files['file']

        if upload_image.filename!='':
            #fname=upload_image.filename
            filepath1=os.path.join(app.config["UPLOAD_FOLDER"],'org.jpg')
            upload_image.save(filepath1)
            if os.path.exists(filepath1):
                return redirect(url_for('processing'))
        else:
            return redirect(url_for('home'))
    return render_template('upload.html')

# Processing route
@app.route('/processing')
def processing():
    return render_template('processing.html')

# Results route
@app.route('/results')
def results():
    imgPath = 'static/images/org.jpg'
    model = YOLO('best.pt')
    results = model.predict('static/images/org.jpg', conf=0.25, iou=0.4)
    transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    image = Image.open(imgPath)  
    input_tensor = transform(image).unsqueeze(0)
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            best_idx = boxes.conf.argmax()
            best_box = boxes[best_idx]
            x_min, y_min, x_max, y_max = best_box.xyxy[0].tolist()
            class_id = best_box.cls.item()
            conf = best_box.conf.item()
        
            # Extract tumor size (in pixels)
            tumor_width_pixels = x_max - x_min
            tumor_height_pixels = y_max - y_min
        
            # Assuming each pixel represents a specific physical size (e.g., 0.5mm per pixel)
            pixel_spacing_mm = 0.5
            tumor_width_mm = tumor_width_pixels * pixel_spacing_mm
            tumor_height_mm = tumor_height_pixels * pixel_spacing_mm
        
            tumor_width_cm = tumor_width_mm / 10
            tumor_height_cm = tumor_height_mm / 10
        
            # Determine tumor type and stage (custom logic can be applied here)
            tumor_type = ''
            if class_id == 0:
                tumor_type = "Glioma"
            elif class_id == 1:
                tumor_type = "Meningioma"
            elif class_id == 2:
                tumor_type = "Pituitary tumor"
            elif class_id == 3:
                tumor_type = "Acoustic neuroma"

    #saving only
    img = cv2.imread(imgPath)  # Replace with your image path
    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
    cv2.putText(img, f'{tumor_type}', (int(x_min), int(y_min) - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
    # Convert image from BGR to RGB for display
    clr1=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static\\images\\results.jpg', results[0].plot(labels=False))                                       #save results[] in storage to acces in this function
    #res=cv2.imread('static\\images\\results.jpg')
    tmvol = 12.5
    # In a real app, this would pass the results data to the template
    return render_template('results.html',size=12.5,volume=tmvol,type=tumor_type)

# History route
@app.route('/history')
def history():
    return render_template('history.html')

# Support route
@app.route('/support')
def support():
    return render_template('support.html')

# Profile route
@app.route('/profile')
def profile():
    return render_template('profile.html')

# Credits route
@app.route('/credits')
def credits():
    return render_template('credits.html')

if __name__ == '__main__':
    app.run(debug=True)

