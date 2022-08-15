from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

import os
from dotenv import load_dotenv

from database import (
    init_db_from_scratch,
    update_db,
    query_recent_asset_data,
    query_recent_model_weights, 
    session,
)

# Get env variables
load_dotenv()

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

# API endpoints
# ==========================
@app.post("/update/")
async def root():
    """Execute the database update method"""
    try:
        dict_info_exec = update_db()
    except:
        dict_info_exec = {'execution':'ERROR'}
    
    return dict_info_exec

@app.post("/initialize/")
async def root():
    """Execute the database initialization method"""
    try:
        init_db_from_scratch()
        dict_info_exec = {'execution':'OK'}
    except:
        dict_info_exec = {'execution':'ERROR'}
    
    return dict_info_exec
