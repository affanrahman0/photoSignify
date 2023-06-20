import requests

# Specify the image file path
image_path = 'NFI-00102014.png'

# Send a POST request to the API endpoint
url = 'http://localhost:5000/predict'
files = {'image': open(image_path, 'rb')}
response = requests.post(url, files=files)

# Get the prediction from the response
str = response.json()['prediction']
if(str>0.5):
    print("Prediction: signature")
else:
    print("Prediction: face")

