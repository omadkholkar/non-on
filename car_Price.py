import flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model_en = pickle.load(open("model_En.pkl", "rb"))
trans_en = pickle.load(open("transmission_En.pkl", "rb"))
fuel_en= pickle.load(open("fuelType_En.pkl", "rb"))
std_scal= pickle.load(open("standard_scaling.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

#@flask_app.route("/")
#def Home():
 #   return render_template("car_price.html")
@flask_app.route('/')
def index():
    my_list = [' A1', ' A6', ' A4', ' A3', ' Q3', ' Q5', ' A5', ' S4', ' Q2',
       ' A7', ' TT', ' Q7', ' RS6', ' RS3', ' A8', ' Q8', ' RS4', ' RS5',
       ' R8', ' SQ5', ' S8', ' SQ7', ' S3', ' S5', ' A2', ' RS7',
       ' 5 Series', ' 6 Series', ' 1 Series', ' 7 Series', ' 2 Series',
       ' 4 Series', ' X3', ' 3 Series', ' X5', ' X4', ' i3', ' X1', ' M4',
       ' X2', ' X6', ' 8 Series', ' Z4', ' X7', ' M5', ' i8', ' M2',
       ' M3', ' M6', ' Z3', ' C Class', ' Focus', 'Focus', ' Fiesta',
       ' Puma', ' Kuga', ' EcoSport', ' C-MAX', ' Mondeo', ' Ka+',
       ' Tourneo Custom', ' S-MAX', ' B-MAX', ' Edge', ' Tourneo Connect',
       ' Grand C-MAX', ' KA', ' Galaxy', ' Mustang',
       ' Grand Tourneo Connect', ' Fusion', ' Ranger', ' Streetka',
       ' Escort', ' Transit Tourneo', ' I20', ' Tucson', ' I10', ' IX35',
       ' I30', ' I40', ' Ioniq', ' Kona', ' Veloster', ' I800', ' IX20',
       ' Santa Fe', ' Accent', ' Terracan', ' Getz', ' Amica', ' SLK',
       ' S Class', ' SL CLASS', ' G Class', ' GLE Class', ' GLA Class',
       ' A Class', ' B Class', ' GLC Class', ' E Class', ' GL Class',
       ' CLS Class', ' CLC Class', ' CLA Class', ' V Class', ' M Class',
       ' CL Class', ' GLS Class', ' GLB Class', ' X-CLASS', '180', ' CLK',
       ' R Class', '230', '220', '200', ' Octavia', ' Citigo',
       ' Yeti Outdoor', ' Superb', ' Kodiaq', ' Rapid', ' Karoq',
       ' Fabia', ' Yeti', ' Scala', ' Roomster', ' Kamiq', ' GT86',
       ' Corolla', ' RAV4', ' Yaris', ' Auris', ' Aygo', ' C-HR',
       ' Prius', ' Avensis', ' Verso', ' Hilux', ' PROACE VERSO',
       ' Land Cruiser', ' Supra', ' Camry', ' Verso-S', ' IQ',
       ' Urban Cruiser', ' Corsa', ' Astra', ' Viva', ' Mokka',
       ' Mokka X', ' Crossland X', ' Zafira', ' Meriva', ' Zafira Tourer',
       ' Adam', ' Grandland X', ' Antara', ' Insignia', ' Ampera', ' GTC',
       ' Combo Life', ' Vivaro', ' Cascada', ' Kadjar', ' Agila',
       ' Tigra', ' Vectra', ' T-Roc', ' Golf', ' Passat', ' T-Cross',
       ' Polo', ' Tiguan', ' Sharan', ' Up', ' Scirocco', ' Beetle',
       ' Caddy Maxi Life', ' Caravelle', ' Touareg', ' Arteon', ' Touran',
       ' Golf SV', ' Amarok', ' Tiguan Allspace', ' Shuttle', ' Jetta',
       ' CC', ' California', ' Caddy Life', ' Caddy', ' Caddy Maxi',
       ' Eos', ' Fox']
    return render_template('car_price.html', my_list=my_list)



@flask_app.route("/car_price", methods = ["GET","POST"])
def predict():
    features = [x for x in request.form.values()]
    
    m =model_en.transform([features[0]])
    features[0]=m[0]
    t =trans_en.transform([features[2]])
    features[2]=t[0]
    f =fuel_en.transform([features[4]])
    features[4]=t[0]
    float_features = [features]
    features =std_scal.transform(float_features)   
    prediction = model.predict(features)
    return render_template("car_price.html", result = "Price of car will be {} $".format(prediction[0]))

if __name__ == "__main__":
    flask_app.run(host='0.0.0.0', port=8501)
