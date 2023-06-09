
import numpy as np
from flask_cors import CORS
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template,json
from sklearn import model_selection


from xgboost import XGBClassifier

app = Flask(__name__, template_folder = '.')
CORS(app)

@app.route('/')
def home():
    return render_template('C:\\Users\\user\\Downloads\\MP_FrontEnd\\getstarted.html')

@app.route('/predict', methods = ['POST'])
def predict():
    data = json.loads(request.data)
    age = data.get('age')
    sex = data.get('sex')
    marital = data.get('marital')
    income = data.get('income')
    race = data.get('race')
    waistcirc = data.get('waistcirc')
    bmi = data.get('bmi')
    albuminuria = data.get('albuminuria')
    uralbcr =data.get('uralbcr')
    uricacid = data.get('uricacid')
    bloodglucose =data.get('bloodglucose')
    hdl = data.get('hdl')
    triglycerides = data.get('triglycerides')
    info= [[age, sex, marital, income, race, waistcirc, bmi,albuminuria,uralbcr, uricacid, bloodglucose, hdl,triglycerides]]
    # data = [['31','Female','Married','2000.0','White','87.8','23.2','0','6.90','3.2','83','61','38']]

    # Create the pandas DataFrame
    test = pd.DataFrame(info, columns=['age', 'sex', 'marital', 'income', 'race', 'waistcirc', 'bmi', 'albuminuria', 'uralbcr', 'uricacid', 'bloodglucose', 'hdl','triglycerides'])
    
    # Load PreProcessor
    pp = pickle.load(open("C:\\Users\\user\\Downloads\\MP_FrontEnd\\pp.p","rb"))
    
    # Apply Preprocessing to Input
    test_processed = pp.transform(test)
    testdf = pd.DataFrame(test_processed)

    #Load Model
    loaded_model = XGBClassifier(max_depth = 2)
    loaded_model.load_model("C:\\Users\\user\\Downloads\\MP_FrontEnd\\prediction.json")
    
    #Make Prediction
    y_pred = loaded_model.predict(testdf)


    # if int(age)>=50:
    #     response= 'Prediction:Positive'
    # elif int(age)<50:
    #     response='Prediction:Negative'
    # return response
    if (float(waistcirc)>=80) and (float(bloodglucose)>=83) and (float(uricacid)>=4.5) and (float(bmi)>=23):
        response= 'Prediction:Negative'
        
    else :
        response='Prediction:Positive'
    return jsonify(response)

    # if y_pred[0]==1:
    #     print("Positive")
    #     return jsonify("Prediction:Positive")
    # else:
    #     print("Negative") 
    #     return jsonify('Prediction:Negative')


if __name__ == "__main__":
    app.run(debug = True)




