import requests

# %% URL of the prediction endpoint
url = 'http://localhost:9696/predict'

# %% Define the data to be sent for prediction
data = {
    "t1": 10.5,
    "t2": 12.5,
    "hum": 78.5,
    "wind_speed": 24.0,
    "weather": "cloudy",
    "is_holiday": "non-holiday",
    "is_weekend": "weekday",
    "season": "winter",
    "month": 1,
    "day": 3,
    "hour": 13,
    "year": 2017,
    "weekday": "Sunday"
}

# %% 
response = requests.post(url, json=data).json()

# %% Print the response 
print(response)

