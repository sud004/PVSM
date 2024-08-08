from fastapi import FastAPI , File , UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image 
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model('../SModel/model.h5')

CLASS_NAME = ['Pepper_bell__Bacterial_spot','Pepper_bell__healthy',
              'Potato__Early_blight','Potato__Late_blight','Potato__healthy',
              'Tomato__Bacterial_spot','Tomato__Early_blight',
              'Tomato__Late_blight','Tomato__Leaf_Mold',
              'Tomato__Septoria_leaf_spot',
              'Tomato__Spider_mites_Two_spotted_spider_mite',
              'Tomato__Target_Spot',
              'Tomato__Tomato_YellowLeaf_Curl_Virus',
              'Tomato__Tomato_mosaic_virus',
              'Tomato__Healthy']




def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get('/hello')
async def ping():
    return 'Hello'


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image  = read_file_as_image(await file.read())
    image_batch  = np.expand_dims(image,0)
    prediction = model.predict(image_batch)
    predicted_class =  CLASS_NAME[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class' : predicted_class,
        'confidence' : float(confidence)

    }



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=9000)