import reflex as rx
from pydantic import BaseModel
import asyncio
import base64
import httpx
import json
import os
import re
import time
import traceback
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")

LOCAL_VLM_URL = os.getenv("LOCAL_VLM_URL", "http://localhost:2023")
ORANGEPI_URL = os.getenv("ORANGEPI_URL", "http://localhost:2022")

# Таймаут на submit (CV: детекция + OCR — секунды/десятки секунд)
ORANGEPI_CV_TIMEOUT = float(os.getenv("ORANGEPI_CV_TIMEOUT", 120.0))
# Таймаут на один poll-запрос (лёгкий, должен отвечать мгновенно)
ORANGEPI_POLL_TIMEOUT = float(os.getenv("ORANGEPI_POLL_TIMEOUT", 30.0))
# Интервал опроса статуса LLM
ORANGEPI_POLL_INTERVAL = float(os.getenv("ORANGEPI_POLL_INTERVAL", 2.0))

BPMN_ANALYSIS_PROMPT = """
Ты — эксперт по анализу BPMN-диаграмм и информационной безопасности.

Проанализируй предоставленное изображение BPMN-диаграммы и верни ответ СТРОГО в виде JSON-объекта без markdown-разметки (без ```json), без пояснений, только сырой JSON.

Для распознавания элементов используй СТРОГО следующий список классов (как в модели детекции):
- arrow_end (конец стрелки/поток управления)
- data_base (база данных/хранилище)
- data_object (объект данных)
- exclusive_gateway (эксклюзивный шлюз, XOR)
- finish_event (конечное событие)
- inclusive_gateway (инклюзивный шлюз, OR)
- intermediate_event (промежуточное событие)
- parallel_gateway (параллельный шлюз, AND)
- role (название процесса ИЛИ роль участника. Это заголовки пулов и дорожек. Рамка должна охватывать весь прямоугольник заголовка вместе с текстом)
- start_event (стартовое событие)
- task (задача/действие)

Структура ответа:
{
  "diagram_type": "<тип диаграммы: BPMN Process Diagram / Collaboration / Choreography / другое>",
  "detected_elements": [
    "<класс_элемента: текст внутри, например: 'start_event: Запрос получен', 'task: Оплата', 'role: Бухгалтерия'>",
    "..."
  ],
  "relationships": [
    "<связь в формате 'Источник → Цель', например: 'start_event → task: Оплата'>",
    "..."
  ],
  "bounding_boxes": [
    {
      "class": "<строго одно из названий классов из списка выше>",
      "label": "<распознанный текст внутри элемента, если есть>",
      "x": 0.0,
      "y": 0.0,
      "w": 0.1,
      "h": 0.1
    }
  ],
  "step_by_step_description": [
    "<шаг 1: подробное описание>",
    "..."
  ],
  "security_issues": [
    "<проблема безопасности 1>",
    "..."
  ]
}

Правила для bounding_boxes:
- x, y — нормализованные координаты левого верхнего угла элемента (0.0–1.0)
- w, h — нормализованные ширина и высота (доля от размеров изображения)
- Укажи bounding box для каждого элемента из detected_elements.
- ВАЖНО для класса 'role': этот класс является общим для названий ролей (дорожек) и названий процессов. Bounding box для 'role' должен строго обводить прямоугольный блок (заголовок пула/дорожки), в котором написан соответствующий текст.

Правила для security_issues:
- Перечисли архитектурные проблемы безопасности, видимые в диаграмме: отсутствие аутентификации/авторизации, незащищённые потоки данных, отсутствие обработки ошибок, потенциальные DoS-векторы, нарушения принципа наименьших привилегий и т.д.
- Если явных проблем нет — верни пустой список [].

Верни ТОЛЬКО JSON-объект.
"""


def _parse_model_json(text: str) -> Dict[str, Any]:
    """Извлекает JSON из ответа модели, устойчив к markdown-обёрткам."""
    text = text.strip()
    text = re.sub(r"^
