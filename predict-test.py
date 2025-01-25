import requests

host = 'share-serving-env.eba-rasgmvfr.ap-southeast-2.elasticbeanstalk.com'
url = f'http://{host}/predict'

data = {
    "t1": 3.5,
    "t2": 9.5,
    "hum": 78.5,
    "wind_speed": 6.0,
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

response = requests.post(url, json=data).json()

print(response)

