import torch
from fastai.learner import load_learner
from PIL import ImageFile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from fastai.imports import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


defaults.device = torch.device('cpu')
learn = load_learner('export.pkl')


def predict(path):
    """Make prediction for image from url, show image and predicted probability."""
    img = open_image(path)
    pred_class, pred_idx, outputs = learn.predict(img)
    print("Predicted Class: ", pred_class)
    print(f"Probability: {outputs[pred_idx].numpy() * 100:.2f}%")
    return pred_class, outputs[pred_idx].numpy() * 100


# some CONST
UPLOAD_FOLDER = 'uploads'

# init and config
app = Flask(__name__)

app.secret_key = "12345"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/food-predict', methods=['POST'])
def food_predict():
    headers = request.headers
    auth = headers.get("X-Api-Key")
    if auth != '123456':
        return jsonify({"message": "ERROR: Unauthorized"}), 401
    try:
        file = request.files['file']
        filename = secure_filename(file.filename)
        content_path = '/uploads/' + filename
        file.save(content_path)
        pred_class, prob = predict(content_path)

        return jsonify(success=True, res=str(pred_class))
    except Exception as e:
        return jsonify(success=False, error=str(e))


if __name__ == "__main__":
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
