from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import logging
from typing import List, Optional
import uvicorn
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели
try:
    with open("movies.joblib", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")

except Exception as e:
    logger.error(f"Error loading model: {e}")
    # raise

with open("power.joblib", 'rb') as file:
    predict2price = pickle.load(file)


app = FastAPI(title="Populary Movie")


def clear_data(df):
    cat_columns = ["original_title", "overview"]
    # num_columns = ['Year', 'Distance', 'Engine_capacity(cm3)', 'Price(euro)']
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    return df

def featurize(df):
    """
        Генерация новых признаков
    """
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day'] = df['release_date'].dt.day
    df['release_weekday'] = df['release_date'].dt.weekday
    df['is_weekend_release'] = df['release_weekday'].isin([5, 6]).astype(int)

    df = df.drop(columns=['release_date'])

    df['years_since_release'] = 2023 - df['release_year']
    df['overview'] = df['overview'].fillna('').astype(str)
    df['overview_length'] = df['overview'].str.len()
    df['vote_power'] = df['vote_average'] * np.log1p(df['vote_count'])
    df['rating_power'] = df['vote_average'] * np.log1p(df['vote_count'])

    # df = df[['popularity', 'vote_average', 'vote_count', 'original_title',
    #          'release_year', 'release_month', 'release_day', 'release_weekday',
    #          'is_weekend_release', 'years_since_release', 'overview_length',
    #          'title_length', 'vote_power', 'rating_power']]

    return df

# Модель входных данных
class MovieFeatures(BaseModel):
    title: str
    overview: str
    release_date: str 
    vote_average: float
    vote_count: int
    
@app.post("/predict", summary="Predict movie popularity")
async def predict(movie: MovieFeatures):
    """
    Предсказывает популярность фильма
    """
    try:
        columns_names = ["original_title", "overview", "release_date", "vote_average", "vote_count"]
        input_data = pd.DataFrame([movie.dict()])
        input_data.columns = columns_names
        featurize_df = featurize(clear_data(input_data))
        print(featurize_df)
        predict = model.predict(featurize_df)[0]
        popularity = predict2price.inverse_transform(predict.reshape(-1,1))
        logger.info(f"Predicted popularity: {popularity}")
        
        return {"predicted_popularity": round(float(popularity), 2)}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)