from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_model_info():
    return {"message": "Model information and performance metrics will be exposed here."}
