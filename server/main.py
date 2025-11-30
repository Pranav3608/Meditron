from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middlewares.exception_handlers import catch_exception_middleware

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
 