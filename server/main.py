from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middlewares.exception_handlers import catch_exception_middleware
from routes.upload_pdf import router as upload_router
from routes.ask_question import router as ask_router

app = FastAPI(title = "Meditron", description = "Medical chatbot")

# Setting up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = ["*"],
    allow_methods =  ["*"],
    allow_headers = ["*"]
)


# Middleware exception handlers
app.middleware("http")(catch_exception_middleware)





# routers
app.include_router(upload_router)
app.include_router(ask_router)