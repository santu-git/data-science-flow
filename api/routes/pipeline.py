from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_pipeline_info():
    return {"message": "Pipeline information and status will be exposed here."}
