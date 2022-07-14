# import requirements needed
from flask import Flask, render_template
from utils import get_base_url
import pandas as pd
import pickle
from flask import request


# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12349
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')


def encode(values):
    df = pd.DataFrame(values).T
    df.columns = ['Age', 'Gender', 'Family_Diabetes', 'highBP', 'PhysicallyActive', 'BMI',
       'Smoking', 'Alcohol', 'Sleep', 'SoundSleep', 'RegularMedicine',
       'JunkFood', 'Stress', 'BPLevel', 'Pregancies', 'Pdiabetes',
       'UriationFreq']
    df = df.astype({"BMI":"int","Sleep":"int","SoundSleep":"int","Pregancies":"int"})
    col_category = ['Age','Gender','Family_Diabetes','highBP','PhysicallyActive','Smoking','Alcohol','RegularMedicine',
                'JunkFood','Stress','BPLevel','Pdiabetes','UriationFreq']
    for col in col_category:
        df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_')], axis=1)
    all_cols = ['BMI','Sleep','SoundSleep','Pregancies','Age_40-49','Age_50-59','Age_60 or older','Age_less than 40','Gender_Female',
                'Gender_Male','Family_Diabetes_no','Family_Diabetes_yes','highBP_no','highBP_yes',
                'PhysicallyActive_less than half an hr','PhysicallyActive_more than half an hr','PhysicallyActive_none',
                'PhysicallyActive_one hr or more','Smoking_no','Smoking_yes','Alcohol_no','Alcohol_yes','RegularMedicine_no','RegularMedicine_o',
                'RegularMedicine_yes','JunkFood_always','JunkFood_occasionally','JunkFood_often','JunkFood_very often','Stress_always',
                'Stress_not at all','Stress_sometimes','Stress_very often','BPLevel_High','BPLevel_Low','BPLevel_high','BPLevel_low','BPLevel_normal',
                'BPLevel_normal ','Pdiabetes_1','Pdiabetes_0','UriationFreq_not much','UriationFreq_quite often']
    for i in all_cols:
        if i not in df:
            df[i] = 0
    df = df[all_cols]
    print(df)
    return df




@app.route(f'{base_url}'  , methods = ["GET","POST"])
def home():
    if request.method == "POST":
        values = [i for i in request.form.values()]
        test = encode(values)
        filename = 'finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(test)
        print(result)
        if result == [1]:
            prediction = 'Diagnosed as diabetic.'
        else:
            prediction = 'Not diagnosed as diabetic.'
        return render_template('index.html', prediction = prediction)


    return render_template('index.html')

# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc19.ai-camp.dev'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
