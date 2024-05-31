import pickle
from pydoc import locate
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

file_name = "xgb_regressor.pkl"
model = pickle.load(open(file_name, "rb"))


def get_features_dict(model):
    feature_names = model.get_booster().feature_names
    feature_types = [Union[float, int] for _ in feature_names]
    return dict(zip(feature_names, feature_types))


def create_input_features_class(model):
    return type("InputFeatures", (BaseModel,), get_features_dict(model))


InputFeatures = create_input_features_class(model)
app = FastAPI()


@app.post("/predict", response_model=List)
async def predict_post(datas: List[InputFeatures]):
    return model.predict(np.asarray([list(data.__dict__.values()) for data in datas])).tolist()
