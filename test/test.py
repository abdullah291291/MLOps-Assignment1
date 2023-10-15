import pickle
import pytest


# print("Hello World")

def load_model():
    model = pickle.load("model.pkl")
    return model

def get_test_data():
    # test_dict = {
    #     'bedrooms':,
    #     'bathrooms':,
    #     'sqft_living':,
    #     'sqft_lot':,
    #     'floors':,
    #     'waterfront':,
    #     'view':,
    #     'condition':,
    #     'sqft_above':,
    #     'sqft_basement':
    # }
    
    test_data = [3.0,1.5,1340,7912,1.5,0,0,3,1340,0]
    return test_data

def test_model():
    y = get_test_data()
    model = load_model()
    
    preds = model.predict(y)
    
    assert len(preds) == len(y)
    
    for pred in preds:
        assert pred > 10000.0

