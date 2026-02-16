from flask import Flask, render_template, request
import pandas as pd
from cyclone_predictor import CycloneDataProcessor, CyclonePredictor, make_future_predictions  # Import from your cyclone_predictor.py

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        year = int(request.form['year'])
        
        processor = CycloneDataProcessor('cyclone.csv')
        processed_df = processor.process_data()
        
        X = processed_df[['Year', 'Month_Num', 'Historical_Avg', 'Trend', 'Volatility', 'Seasonal_Strength']]
        
        y_dict = {
            'BOB': (processed_df['BOB'] > 0).astype(int),
            'AS': (processed_df['AS'] > 0).astype(int),
            'LAND': (processed_df['LAND'] > 0).astype(int)
        }
        
        predictor = CyclonePredictor(X, y_dict)
        rf_models, scaler = predictor.train_models()
        
        predictions_df = make_future_predictions(year, processed_df, rf_models, scaler)
        
        return render_template('index.html', predictions=predictions_df.to_html(classes='table table-striped'), year=year)
    
    return render_template('index.html', predictions=None)

if __name__ == '__main__':
    app.run(debug=True)
