import uvicorn
from fastapi import FastAPI, File, UploadFile
from prediction import predict
from PIL import Image
app = FastAPI()

# Corrected function for reading the image file
def read_imagefile(file):
    image = Image.open(file.file)
    return image

@app.post("/predict/image")
async def predict_api(file: UploadFile):
    image = read_imagefile(file)
    extension = file.filename.split(".")[-1].lower()
    if extension not in ("jpg", "jpeg", "png"):
        return "Image must be jpg, jpeg, or png format!"
    
    # Corrected function call to predict
    prediction = predict(image)
    return prediction

