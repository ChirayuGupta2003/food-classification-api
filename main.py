from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
import os
import cv2
import gdown


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def predict_image(img_array):

    img_processed = np.expand_dims(img_array, axis=0)
    img_processed = img_processed / 255

    prediction = m.predict(img_processed)

    index = np.argmax(prediction)
    confidence = prediction[0][index]

    return index, confidence


@app.route("/")
def home():
    file = request.files['file'].read()

    npimg = np.frombuffer(file, np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (228, 228), interpolation=cv2.INTER_NEAREST)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    index, confidence = predict_image(img)

    data = {"class": classes[index], "confidence": str(confidence)}
    print(data)

    if data:
        return jsonify(data)
    return "Working"


if __name__ == "__main__":

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

    app.run(debug=False, host="0.0.0.0")
