# Create a file main.py 
# Use FastAPI to create a simple API with two endpoints: 
# one for the root and another for retrieving items by ID.
from fastapi import FastAPI, Request,Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.predict import predict_review

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.post("/predict")
async def predict(request: Request, review: str = Form(...)):
    prediction, prob = predict_review(review)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "review": review, 
            "prediction": prediction, 
            "probability": prob
            }
    )