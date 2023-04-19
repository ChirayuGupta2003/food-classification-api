import flask
import flask_cors
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
import gdown
import PIL


app = flask.Flask(__name__)
cors = flask_cors.CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def predict_image(filename, model):
    img_ = image.load_img(filename, target_size=(228, 228))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)

    index = np.argmax(prediction)

    return index, prediction[0][index]


@app.route("/")
def home():
    return flask.render_template("index.html")


@app.route("/predict")
def predict():
    try:
        file = flask.request.files['file'].read()
    except KeyError as e:
        return f"No image file attached<br>{e}"

    npimg = np.frombuffer(file, np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    cv2.imwrite("image.jpg", img)

    index, confidence = predict_image("image.jpg", m)

    data = {"class": classes[index], "confidence": str(confidence)}
    print(data)

    os.remove('image.jpg')

    if data:
        return data
    return "Working"


model = InceptionV3(weights='imagenet', include_top=False,
                    input_shape=(228, 228, 3), pooling='avg')

for layer in model.layers:
    layer.trainable = False

input = model.output
input = BatchNormalization(axis=-1)(input)
input = Dense(1024, activation='relu')(input)
input = Dropout(0.3)(input)
output = Dense(20, activation='softmax')(input)
m = Model(inputs=model.input, outputs=output)
m.compile(optimizer='adam', metrics=[
    'accuracy'], loss='categorical_crossentropy')

if not os.path.isfile("food_recognition_inceptionV3.h5"):
    gdown.download("https://drive.google.com/file/d/1Y5ALEolZrlyYYRx9HEIC39aU4eBV2XjF/view?usp=share_link",
                   "food_recognition_inceptionV3.h5", quiet=False, fuzzy=True)

m.load_weights("food_recognition_inceptionV3.h5")

classes = ['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi',
           'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa']


if __name__ == "__main__":
    app.run(debug=True, port=5000)
