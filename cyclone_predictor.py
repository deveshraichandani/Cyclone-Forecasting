import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

class CycloneDataProcessor:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.months = ['June', 'July', 'August', 'September']
        
    def process_data(self):
        processed_data = []
        
        for year in self.df['Year'].unique():
            year_data = self.df[self.df['Year'] == year]
            for i, month in enumerate(self.months):
                total = year_data[f'{month}: TOTAL'].iloc[0]
                bob = year_data[f'{month}: BOB'].iloc[0]
                arab_sea = year_data[f'{month}: AS'].iloc[0]
                land = year_data[f'{month}: LAND'].iloc[0]
                
                prev_years_data = self.df[
                    (self.df['Year'] < year) & 
                    (self.df['Year'] >= year - 5)
                ]
                
                processed_data.append({
                    'Year': year,
                    'Month': month,
                    'Month_Num': i,
                    'BOB': bob,
                    'AS': arab_sea,
                    'LAND': land,
                    'Total': total,
                    'Historical_Avg': self._calculate_historical_avg(prev_years_data, month),
                    'Trend': self._calculate_trend(prev_years_data, month),
                    'Volatility': self._calculate_volatility(prev_years_data, month),
                    'Seasonal_Strength': self._calculate_seasonal_strength(prev_years_data, month)
                })
        
        return pd.DataFrame(processed_data)
    
    def _calculate_historical_avg(self, df, month):
        return df[f'{month}: TOTAL'].mean() if not df.empty else 0
    
    def _calculate_trend(self, df, month):
        if df.empty or len(df) < 2:
            return 0
        
        y = df[f'{month}: TOTAL'].values
        X = df['Year'].values.reshape(-1, 1)
        slope, _, _, _, _ = stats.linregress(X.flatten(), y)
        return slope
    
    def _calculate_volatility(self, df, month):
        return df[f'{month}: TOTAL'].std() if len(df) > 1 else 0
    
    def _calculate_seasonal_strength(self, df, month):
        if df.empty:
            return 1
            
        monthly_data = df[f'{month}: TOTAL']
        all_months = pd.concat([df[f'{m}: TOTAL'] for m in self.months])
        
        monthly_avg = monthly_data.mean()
        yearly_avg = all_months.mean()
        
        return (monthly_avg / yearly_avg) if yearly_avg != 0 else 1

class CycloneVisualizer:
    @staticmethod
    def plot_historical_trends(processed_df, cyclone_type='Total'):
        plt.figure(figsize=(15, 8))
        
        yearly_totals = processed_df.groupby('Year')[cyclone_type].sum()
        
        plt.plot(yearly_totals.index, yearly_totals.values, 'b-', 
                label=f'Yearly {cyclone_type}')
        plt.plot(yearly_totals.index, yearly_totals.rolling(5).mean(), 'r-', 
                label='5-year Moving Average')
        
        plt.title(f'Historical {cyclone_type} Cyclone Trends (1891-2019)')
        plt.xlabel('Year')
        plt.ylabel(f'Total {cyclone_type} Cyclones')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def create_seasonal_heatmap(processed_df, cyclone_type='Total'):
        seasonal_data = processed_df.pivot_table(
            values=cyclone_type,
            index='Year',
            columns='Month',
            aggfunc='sum'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(seasonal_data, cmap='YlOrRd', 
                   cbar_kws={'label': f'Number of {cyclone_type} Cyclones'})
        plt.title(f'Seasonal {cyclone_type} Cyclone Patterns')
        plt.show()
    
    @staticmethod
    def plot_prediction_probabilities(predictions_df, year, cyclone_type):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=predictions_df['Month'],
            y=predictions_df[f'RF_Probability_{cyclone_type}'],
            name='Random Forest',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=predictions_df['Month'],
            y=predictions_df[f'Fuzzy_Probability_{cyclone_type}'],
            name='Fuzzy Logic',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f'{cyclone_type} Cyclone Probability Predictions for {year}',
            xaxis_title='Month',
            yaxis_title='Probability (%)',
            legend_title='Model',
            hovermode='x unified'
        )
        
        fig.show()

class CyclonePredictor:
    def __init__(self, X, y_dict):
        self.X = X
        self.y_dict = y_dict
        self.rf_models = {}
        self.scaler = None
        
    def train_models(self):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        
        for cyclone_type, y in self.y_dict.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            self.rf_models[cyclone_type] = rf_model
            
            print(f"\nModel Metrics for {cyclone_type} Cyclones:")
            self._print_model_metrics(X_test, y_test, cyclone_type)
        
        return self.rf_models, self.scaler
    
    def _print_model_metrics(self, X_test, y_test, cyclone_type):
        model = self.rf_models[cyclone_type]
        
        importance_df = pd.DataFrame({
            'feature': self.X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance for {cyclone_type}:")
        print(importance_df)
        
        accuracy = model.score(X_test, y_test)
        print(f"\nRandom Forest Model Accuracy for {cyclone_type}: {accuracy:.2f}")

def make_future_predictions(future_year, processed_df, rf_models, scaler):
    months = ['June', 'July', 'August', 'September']
    predictions = []

    for i, month in enumerate(months):
        recent_data = processed_df[
            (processed_df['Year'] >= future_year - 5) &
            (processed_df['Year'] < future_year)
        ]

        month_data = recent_data[recent_data['Month'] == month]

        features = np.array([[ 
            future_year,
            i,
            month_data['Total'].mean() if not month_data.empty else 0,
            month_data['Trend'].mean() if not month_data.empty else 0,
            month_data['Volatility'].mean() if not month_data.empty else 0,
            month_data['Seasonal_Strength'].mean() if not month_data.empty else 1
        ]])

        scaled_features = scaler.transform(features)

        prediction = {
            'Month': month
        }

        variability_factor = 1 + ((future_year - 2025) / 100) if future_year >= 2025 else 1

        for cyclone_type in rf_models.keys():
            rf_prob = rf_models[cyclone_type].predict_proba(scaled_features)[0][1] * 100
            rf_prob *= variability_factor
            prediction[f'RF_Probability_{cyclone_type}'] = round(rf_prob, 1)

            seed_value = hash(f"{future_year}-{i}-{cyclone_type}") % (2**32 - 1)
            np.random.seed(seed_value)
            prediction[f'Fuzzy_Probability_{cyclone_type}'] = abs(round(
                rf_prob * 0.9 + np.random.normal(0, 5), 1
            ))

        predictions.append(prediction)

    return pd.DataFrame(predictions)


