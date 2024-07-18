from flask import Flask , render_template , request
import pandas as pd
import pickle
app = Flask(__name__)

model = pickle.load(open('LinearRegression.pkl' , 'rb'))
data  = pd.read_csv('Cleaned_data.csv')

@app.route('/')
def index():
    companies = sorted(data['company'].unique())
    car_models = sorted(data['name'].unique())
    years = sorted(data['year'].unique() , reverse=True)
    fuel_type = sorted(data['fuel_type'].unique())
    companies.insert(0 , 'Select Comapany')
    return render_template('index.html' , companies=companies , car_models=car_models , years=years ,
                           fuel_types=fuel_type)
                           
@app.route('/predict' , methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model=request.form.get('car_model')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    kms_driven=int(request.form.get('kilo_driven'))
    prediction = model.predict(pd.DataFrame([[car_model , company , year , kms_driven , fuel_type]] , 
                                            columns=['name' , 'company' , 'year' , 'kms_driven' , 'fuel_type']))    
    return str(round(prediction[0 , 0] , 2))
if __name__ == '__main__':
    app.run(debug=True)