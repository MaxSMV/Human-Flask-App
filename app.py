from flask import Flask, request, jsonify, make_response
import mlflow
import mlflow.tensorflow
import numpy as np
import cv2
import os
import base64

app = Flask(__name__)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Set MLflow Tracking URI
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

# Load MLflow model
model_uri = "runs:/11ea479c5a554b5286c68ceedd61754f/emotion_detection_model"
model = mlflow.tensorflow.load_model(model_uri)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_data = request.form['img_data']
        img_data = base64.b64decode(img_data)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find haar cascade to draw bounding box around face
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi_gray_img = gray_img[y:y + h, x:x + w]
            roi_gray_img = cv2.resize(roi_gray_img, (48, 48))
            roi_gray_img = roi_gray_img.reshape(1, 48, 48, 1)
            roi_gray_img = roi_gray_img.astype('float32') / 255

            # Start a MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
            with mlflow.start_run() as run:
                predictions = model.predict(roi_gray_img)
                maxindex = int(np.argmax(predictions))
                emotion = emotion_dict[maxindex]

                # Log model performance
                mlflow.log_param("emotion", emotion)

            return jsonify(x=int(x), y=int(y), w=int(w), h=int(h), emotion=emotion)
        else:
            return jsonify(error='No face detected')
    except Exception as e:
        print(f"Error: {e}")
        return make_response(jsonify(error='Error occurred during processing'), 500)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
