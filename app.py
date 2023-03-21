import numpy as np
import pandas as pd
import xgboost as xg
from flask import Flask, render_template, request

app = Flask(__name__)

def Train():
    data = pd.read_csv('Dataset/Electric_Production.csv')
    data['DATE'] = pd.to_datetime(data['DATE'])
    data['Day'] = data['DATE'].dt.day
    data['Month'] = data['DATE'].dt.month
    data['Year'] = data['DATE'].dt.year

    model = xg.XGBRegressor(
        objective ='reg:squarederror',
        n_estimators = 600
    ).fit(
        data[['Day', 'Month', 'Year']],
        data['IPG2211A2N']
    )
    print('Model Trained :-)')
    return model

model = Train()

Test_Data = pd.read_csv('Dataset/TestData.csv')
X_List = []
Y_List = []

count = 0
@app.route('/live', methods=['GET'])
def Live():
    global count 
    global X_List 
    global Y_List
    for i in range(30):
        item = Test_Data.iloc[count][['Day', 'Month', 'Year']].to_numpy().reshape(1,-1)
        out = int(model.predict(item)[0])
        X_List.append(str(Test_Data.iloc[count]['Date']))
        Y_List.append(out)
        count = count+1
        if count==1000:
            X_List = []
            Y_List = []
        if count==1820:
            count=0
            X_List = []
            Y_List = []
            break
    # print(X_List)
    # print(Y_List)
    return render_template(
        'plot.html',
        X=X_List,
        Y=Y_List
    )


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        item = np.array([
            float(request.form['D101']), 
            float(request.form['M101']),
            float(request.form['Y101'])
        ]).reshape(1,-1)

        out = model.predict(item)
        return render_template('predict.html', out=out)
    else:
        return render_template("file.html")

if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0', 
        use_reloader=False
    )