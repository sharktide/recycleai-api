from fastapi import FastAPI, File, UploadFile, Request
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import torch  # Make sure torch is imported
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your trained model
model = tf.keras.models.load_model('recyclebot.keras')

# Load background removal model
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)

# Transform for the background removal model
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define class names for predictions (this should be the same as in your local code)
CLASSES = ['Glass', 'Metal', 'Paperboard', 'Plastic-Polystyrene', 'Plastic-Regular']

# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify specific origins)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Preprocess the image (resize, reshape without normalization)
def preprocess_image(image_file):
    try:
        # Load image using PIL
        image = Image.open(image_file)
        
        # Convert image to numpy array
        image = np.array(image)
        
        # Resize to the input shape expected by the model
        image = cv2.resize(image, (240, 240))  # Resize image to match model input
        
        # Reshape the image (similar to your local code)
        image = image.reshape(-1, 240, 240, 3)  # Add the batch dimension for inference
        
        return image
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

# Background removal function
def remove_background(image):
    try:
        image_size = image.size
        input_images = transform_image(image).unsqueeze(0)
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    except Exception as e:
        logger.error(f"Error in remove_background: {str(e)}")
        raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info("Received request for /predict")
        img_array = preprocess_image(file.file)  # Preprocess the image
        prediction1 = model.predict(img_array)  # Get predictions
        
        predicted_class_idx = np.argmax(prediction1, axis=1)[0]  # Get predicted class index
        predicted_class = CLASSES[predicted_class_idx]  # Convert to class name
        
        return JSONResponse(content={"prediction": predicted_class})
    
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/predict/recyclebot0accuracy")
async def predict_recyclebot0accuracy(file: UploadFile = File(...)):
    try:
        logger.info("Received request for /predict/recyclebot0accuracy")
        # Load and remove background from image
        image = Image.open(file.file).convert("RGB")
        image = remove_background(image)
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image_path = "processed_image.jpg"
        image.save(image_path, "JPEG")
        
        # Preprocess the image with the background removed
        img_array = preprocess_image(image_path)
        
        # Get predictions
        prediction1 = model.predict(img_array)
        
        predicted_class_idx = np.argmax(prediction1, axis=1)[0]  # Get predicted class index
        predicted_class = CLASSES[predicted_class_idx]  # Convert to class name
        
        return JSONResponse(content={"prediction": predicted_class})
    
    except Exception as e:
        logger.error(f"Error in /predict/recyclebot0accuracy: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.get("/working")
async def working():
    return JSONResponse(content={"Status": "Working"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
