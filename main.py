import pickle
from pydoc import locate
from typing import List, Union

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

file_name = "xgb_regressor.pkl"
model = pickle.load(open(file_name, "rb"))

# Obtener nombres de características en el orden esperado por el modelo
feature_names_in_order = model.get_booster().feature_names

def get_features_dict(model):
    features_dict = {}
    for feature_name in feature_names_in_order:
        features_dict[feature_name]: Union[float, int] 
    return features_dict 

def create_input_features_class(model):
    return type("InputFeatures", (BaseModel,), get_features_dict(model))

InputFeatures = create_input_features_class(model)
app = FastAPI()

@app.post("/predict", response_model=List)
async def predict_post(datas: List[InputFeatures]):
    # Construir el array NumPy usando el orden correcto de características
    input_array = np.asarray([[getattr(data, feature) for feature in feature_names_in_order] for data in datas])
    
    print("Forma del array de entrada:", input_array.shape)  # Verificar forma del array
    
    return model.predict(input_array).tolist() 
