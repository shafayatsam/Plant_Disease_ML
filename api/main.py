from fastapi import FastAPI, File, UploadFile
from starlette.requests import Request
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

MODEL = tf.keras.models.load_model('plant_model_save.h5')

CLASS_NAMES = ['Pepper bell Bacterial spot',
                'Pepper bell healthy',
                'Potato Early blight',
                'Potato Late blight',
                'Potato healthy',
                'Tomato Bacterial spot',
                'Tomato Early blight',
                'Tomato Late blight',
                'Tomato Leaf Mold',
                'Tomato Septoria leaf spot',
                'Tomato Spider mites Two spotted spider mite',
                'Tomato Target Spot',
                'Tomato Tomato Yellow Leaf Curl Virus',
                'Tomato Tomato mosaic virus',
                'Tomato healthy']

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/ping")
async def ping(request : Request):
    return templates.TemplateResponse('upload1.html', {'request' : request})


@app.get('/predict')
async def predict(request: Request):
    return templates.TemplateResponse('upload1.html',{'request' : request})


@app.post("/predict")
async def predict(request : Request, file: UploadFile = File(...)):
    
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(image_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    return templates.TemplateResponse('index.html',
        {
        'request': request, 
        'class' : predicted_class,
        'confidence' : round(float(confidence), 2)
        })



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)