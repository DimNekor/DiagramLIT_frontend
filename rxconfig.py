import reflex as rx
import os
from dotenv import load_dotenv

config = rx.Config(
    app_name="DiagramLIT_frontend",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
    # api_url=os.getenv("BACKEND_URL", "http://localhost:8080"),
    api_url="https://diagramlit.ru",
    # api_url="http://localhost:8000",
)
