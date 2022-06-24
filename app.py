from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np

import tensorflow as tf
app = Flask(__name__)

dic =  ['Dense', 'Not Dense', "Not Dense"]
model = load_model('resnet50.h5')

#model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224,224))
    
    i = np.expand_dims(i, axis=0)
    i = tf.convert_to_tensor(i)
    p = model.predict(i, batch_size=1)
    p = np.argmax(p,axis=1)
    return dic[p[0]]



 


    

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)