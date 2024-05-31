import pickle
from pydoc import locate
from typing import List, Union

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

file_name = "xgb_regressor.pkl"
model = pickle.load(open(file_name, "rb"))


def get_features_dict(model):
    feature_names = model.get_booster().feature_names
    features_dict = {}
    for feature_name in feature_names:
        # Asignar anotación de tipo Union[float, int] a cada característica
        features_dict[feature_name] : Union[float, int] 
    return features_dict


def create_input_features_class(model):
    return type("InputFeatures", (BaseModel,), get_features_dict(model))


InputFeatures = create_input_features_class(model)
app = FastAPI()


@app.post("/predict", response_model=List)
async def predict_post(datas: List[InputFeatures]):
    return model.predict(np.asarray([list(data.__dict__.values()) for data in datas])).tolist()
