from flask import Flask, render_template, request, jsonify
from pathlib import Path
import os
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# Create directories
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(parents=True, exist_ok=True)

# ============================================
# MODEL LOADING - SWITCH BETWEEN MOCK AND REAL
# ============================================

MODEL_PATH = Path('models/best.pt')
USE_MOCK = not MODEL_PATH.exists()

if USE_MOCK:
    print("⚠️  Model not found - using MOCK model for testing")
    from mock_yolo import MockYOLO
    model = MockYOLO()
else:
    print("✓ Loading real YOLO model...")
    
    # FIX FOR PYTORCH 2.6+ - Add all necessary safe globals
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn.modules.container import Sequential
    
    # Add all required classes to safe globals
    torch.serialization.add_safe_globals([
        DetectionModel,
        Sequential,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.activation.SiLU,
        torch.nn.modules.pooling.MaxPool2d,
        torch.nn.modules.linear.Linear,
    ])
    
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))

# ============================================
# ROUTES
# ============================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        allowed_ext = {'png', 'jpg', 'jpeg'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_ext):
            return jsonify({'error': 'Only PNG/JPG allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Run inference
        start_time = time.time()
        
        if USE_MOCK:
            result_img, detections = model(upload_path)
        else:
            # Real YOLO model inference
            results = model(upload_path)
            result = results[0]
            
            import cv2
            img = cv2.imread(upload_path)
            result_img = img.copy()
            detections = []
            
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]
                
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{class_name} {conf:.2f}"
                cv2.putText(result_img, text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
        
        inference_time = time.time() - start_time
        
        # Save result image
        result_filename = f"result_{int(time.time())}.jpg"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        import cv2
        cv2.imwrite(result_path, result_img)
        
        return jsonify({
            'success': True,
            'uploaded_image': f"/{upload_path}",
            'result_image': f"/{result_path}",
            'detections': detections,
            'inference_time': f"{inference_time:.2f}s",
            'num_objects': len(detections)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
