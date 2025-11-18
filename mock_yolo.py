import cv2
import random

class MockYOLO:
    """Simulates YOLO model for development without actual model"""
    
    def __call__(self, image_path):
        """Mock inference - returns dummy detections"""
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = img.shape[:2]
        
        # Create random mock detections
        num_detections = random.randint(1, 5)
        mock_boxes = []
        
        classes = ['person', 'car', 'dog', 'cat', 'bird', 'bicycle']
        
        for i in range(num_detections):
            x1 = random.randint(10, max(60, w - 150))
            y1 = random.randint(10, max(60, h - 150))
            x2 = min(x1 + random.randint(80, 200), w)
            y2 = min(y1 + random.randint(80, 200), h)
            
            mock_boxes.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'conf': round(random.uniform(0.75, 0.99), 2),
                'class': random.choice(classes)
            })
        
        # Draw boxes on image
        result_img = img.copy()
        for box in mock_boxes:
            cv2.rectangle(result_img, (box['x1'], box['y1']), (box['x2'], box['y2']), 
                         (0, 255, 0), 2)
            text = f"{box['class']} {box['conf']:.2f}"
            cv2.putText(result_img, text, (box['x1'], box['y1']-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_img, mock_boxes
