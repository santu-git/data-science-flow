from fastapi import FastAPI
from api.routes import model, pipeline

app = FastAPI()

# Include routers
app.include_router(model.router, prefix="/api/model", tags=["Model"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Card Fraud Detection API"}
