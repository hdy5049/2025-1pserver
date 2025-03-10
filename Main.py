import uvicorn
from fastapi import FastAPI

from irisModel import IrisMachineLearning

app = FastAPI()

model = IrisMachineLearning()

@app.get("/")
async def root():
    return {"message": "Hello, this is iris classfier 2025/3/10"}

@app.get("/predict")
async def predict():
    pred = model.predict_species(8,1,8,1)
    return {"prediction":pred}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)