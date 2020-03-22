from flask import Flask, url_for, send_from_directory, request, jsonify
import logging
import os
from werkzeug import secure_filename
from datetime import datetime
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient


name = 'Azure Classification'
app = Flask(name)

file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
ALLOWED = {'image/jpeg','image/jpg', 'image/gif', 'image/bmp', 'image/png'}
ENDPOINT = "https://southeastasia.api.cognitive.microsoft.com/"

prediction_key = ""
prediction_resource_id = ""
project_id = ""
publish_iteration_name = ""
predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return 'Hii'

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

def generate_date():
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")

def check_mime(file_type):
    if file_type in ALLOWED:
        return True
    return False

def predict(img):
    with open(img, "rb") as image_contents:
        results = predictor.classify_image(
        project_id, publish_iteration_name, image_contents.read())
        print(results)
        res =[]
        for prediction in results.predictions:
            print(prediction.tag_name +": {0:.2f}%".format(prediction.probability * 100))
            #res.append(prediction.tag_name +": {0:.2f}%".format(prediction.probability * 100))
            res.append("'{}'".format(prediction.tag_name)+":"+"'{0:.2f}%'".format(prediction.probability * 100))
        return res
        
@app.route('/', methods=['POST'])
def api_root():
    app.logger.info(generate_date()+' init '+PROJECT_HOME)
    now = datetime.now()
    print(now)
    if request.method == 'POST' and request.files['image']:
        app.logger.info(generate_date()+' '+app.config['UPLOAD_FOLDER'])
        img = (request.files['image'])
        if img.filename == '':
            return jsonify({'status': 'Image Not Found'}), 404
        if check_mime(img.content_type) == False:
            app.logger.info(generate_date()+'Issue Detected '+img.filename)
            return jsonify({'status': 'Not Allowed'}), 422
        img_name = secure_filename(generate_date()+'_'+img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("{} saving {}".format(generate_date(), saved_path))
        img.save(saved_path)
        res = predict(saved_path)
        time_elapsed = datetime.now()-now
        #print(time_elapsed)
        return jsonify({'status':'success','data':res,'time':str(time_elapsed.microseconds)+' us'})
    else:
        return jsonify({'status': 500}), 500
