import reflex as rx
import asyncio
from typing import List, Dict, Any


class State(rx.State):
    """Состояние приложения DiagramLIT"""

    # Состояние загрузки
    is_uploading: bool = False
    is_processing: bool = False

    # Данные о диаграмме
    uploaded_filename: str = ""
    diagram_type: str = ""
    detected_elements: List[str] = []
    relationships: List[str] = []
    step_by_step_description: List[str] = []

    # Текущая активная страница
    current_step: int = 0

    async def handle_upload(self, files: List[rx.UploadFile]):
        """Обработка загруженного файла диаграммы"""
        if not files:
            return

        self.is_uploading = True
        self.uploaded_filename = files[0].filename
        yield

        try:
            file_content = await files[0].read()

            # Временная имитация для демонстрации
            await asyncio.sleep(1.5)

            self.diagram_type = "UML Class Diagram"
            self.detected_elements = ["User", "Order", "Product", "Payment"]
            self.relationships = ["User → Order", "Order → Product", "Payment → Order"]
            self.step_by_step_description = [
                "Пользователь создаёт новый заказ",
                "Система проверяет наличие товаров",
                "Формируется счёт на оплату",
                "Пользователь подтверждает платеж",
                "Статус заказа обновляется на 'Оплачен'",
            ]

        except Exception as e:
            print(f"Ошибка: {e}")
            self.diagram_type = "Ошибка"
        finally:
            self.is_uploading = False

        yield rx.redirect("/results")

    def next_step(self):
        if self.current_step < len(self.step_by_step_description) - 1:
            self.current_step += 1

    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1

    def clear_upload(self):
        self.uploaded_filename = ""
        self.diagram_type = ""
        self.detected_elements = []
        self.relationships = []
        self.step_by_step_description = []
        self.current_step = 0


def navbar():
    """Навигационная панель"""
    return rx.hstack(
        rx.hstack(
            rx.image(src="/favicon.ico", width="32px", height="32px"),
            rx.heading("DiagramLIT", size="6", color="white", font_weight="bold"),
            spacing="2",
            align="center",
        ),
        rx.hstack(
            rx.link("Главная", href="/", color="white", font_weight="500"),
            rx.link(
                "GitHub",
                href="https://github.com",
                color="white",
                font_weight="500",
                is_external=True,
            ),
            spacing="4",
        ),
        justify="between",
        align="center",
        padding_x="2em",
        padding_y="1em",
        bg="rgba(255, 255, 255, 0.1)",
        backdrop_filter="blur(10px)",
        box_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1)",
    )


def footer():
    """Подвал"""
    return rx.center(
        rx.vstack(
            rx.text(
                "© 2024 DiagramLIT — AI-powered Diagram Analysis",
                font_size="0.8em",
                color="#a0aec0",
            ),
            rx.text(
                "Распознавание UML, BPMN, Activity, C4 диаграмм",
                font_size="0.7em",
                color="#a0aec0",
            ),
            align="center",
            spacing="1",
        ),
        padding="2em",
        bg="rgba(0, 0, 0, 0.3)",
        width="100%",
    )


def index() -> rx.Component:
    """Главная страница с загрузкой диаграммы"""
    return rx.fragment(
        navbar(),
        rx.container(
            rx.vstack(
                # Hero section - теперь текст белый
                rx.vstack(
                    rx.heading(
                        "Анализ бизнес-диаграмм с помощью ИИ",
                        size="8",
                        text_align="center",
                        font_weight="bold",
                        color="white",  # Белый цвет
                        text_shadow="0 2px 4px rgba(0,0,0,0.2)",
                    ),
                    rx.text(
                        "Загрузите диаграмму — получите пошаговое описание всех элементов",
                        font_size="1.2em",
                        text_align="center",
                        color="#e2e8f0",  # Светло-серый
                    ),
                    spacing="3",
                    align="center",
                    width="100%",
                    padding_bottom="2em",
                ),
                # Upload card - белая карточка
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("upload", size=30, color="#667eea"),
                            rx.heading("Загрузка диаграммы", size="5", color="#1a202c"),
                            spacing="3",
                            align="center",
                        ),
                        rx.text(
                            "Поддерживаемые форматы: PNG, JPG, JPEG, SVG",
                            font_size="0.8em",
                            color="#718096",
                        ),
                        rx.divider(),
                        # Upload area
                        rx.upload(
                            rx.vstack(
                                rx.icon("image", size=40, color="#a0aec0"),
                                rx.text(
                                    "Перетащите файл сюда или",
                                    font_size="0.9em",
                                    color="#4a5568",
                                ),
                                rx.button(
                                    "Выбрать файл",
                                    color_scheme="blue",
                                    size="3",
                                    variant="solid",
                                ),
                                rx.cond(
                                    State.uploaded_filename,
                                    rx.badge(
                                        f"Файл: {State.uploaded_filename}",
                                        color_scheme="green",
                                        variant="soft",
                                        font_size="0.8em",
                                    ),
                                ),
                                spacing="3",
                                align="center",
                            ),
                            on_drop=State.handle_upload,
                            multiple=False,
                            accept={"image/*": [".png", ".jpg", ".jpeg", ".svg"]},
                            border="2px dashed #cbd5e0",
                            border_radius="12px",
                            padding="2.5em",
                            bg="#fafafa",
                        ),
                        # Loading indicator
                        rx.cond(
                            State.is_uploading,
                            rx.center(
                                rx.spinner(size="3", color="#667eea"),
                                rx.text(
                                    "Обработка диаграммы...",
                                    margin_left="1em",
                                    color="#4a5568",
                                ),
                                padding="2em",
                            ),
                        ),
                        # Clear button
                        rx.cond(
                            State.uploaded_filename,
                            rx.button(
                                rx.hstack(
                                    rx.icon("trash-2", size=16),
                                    rx.text("Очистить"),
                                ),
                                on_click=State.clear_upload,
                                color_scheme="red",
                                variant="outline",
                                size="2",
                            ),
                        ),
                    ),
                    spacing="5",
                    align="stretch",
                    width="100%",
                    padding="2em",
                    box_shadow="0 10px 15px -3px rgba(0, 0, 0, 0.1)",
                    border_radius="16px",
                    bg="white",  # Белый фон карточки
                ),
                # Features section - карточки с белым фоном
                rx.grid(
                    rx.card(
                        rx.vstack(
                            rx.icon("brain", size=32, color="#667eea"),
                            rx.heading("ИИ-распознавание", size="4", color="#1a202c"),
                            rx.text(
                                "Автоматическое определение типов диаграмм и их элементов",
                                text_align="center",
                                font_size="0.9em",
                                color="#4a5568",
                            ),
                            spacing="3",
                            align="center",
                        ),
                        padding="1.5em",
                        bg="white",
                        border_radius="12px",
                    ),
                    rx.card(
                        rx.vstack(
                            rx.icon("message-square", size=32, color="#667eea"),
                            rx.heading("Пошаговое описание", size="4", color="#1a202c"),
                            rx.text(
                                "Детальное объяснение всех процессов и связей",
                                text_align="center",
                                font_size="0.9em",
                                color="#4a5568",
                            ),
                            spacing="3",
                            align="center",
                        ),
                        padding="1.5em",
                        bg="white",
                        border_radius="12px",
                    ),
                    rx.card(
                        rx.vstack(
                            rx.icon("layout", size=32, color="#667eea"),
                            rx.heading("Все форматы", size="4", color="#1a202c"),
                            rx.text(
                                "Поддержка UML, BPMN, Activity, C4 диаграмм",
                                text_align="center",
                                font_size="0.9em",
                                color="#4a5568",
                            ),
                            spacing="3",
                            align="center",
                        ),
                        padding="1.5em",
                        bg="white",
                        border_radius="12px",
                    ),
                    columns="repeat(3, 1fr)",
                    gap="6",
                    width="100%",
                    padding_y="3em",
                ),
                spacing="5",
                align="stretch",
            ),
            max_width="1200px",
            padding_x="2em",
            padding_y="3em",
            min_height="calc(100vh - 140px)",
        ),
        footer(),
    )


def results() -> rx.Component:
    """Страница с результатами распознавания"""
    return rx.fragment(
        navbar(),
        rx.container(
            rx.vstack(
                # Header with navigation
                rx.hstack(
                    rx.link(
                        rx.button(
                            rx.hstack(
                                rx.icon("arrow-left", size=16),
                                rx.text("Назад"),
                            ),
                            variant="outline",
                            bg="white",
                        ),
                        href="/",
                    ),
                    rx.heading("Результаты анализа", size="7", color="white"),
                    spacing="4",
                    align="center",
                    width="100%",
                ),
                rx.divider(),
                # Diagram type card
                rx.cond(
                    State.diagram_type,
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("bar-chart-2", size=24, color="#667eea"),
                                rx.heading("Тип диаграммы", size="4"),
                                spacing="2",
                                align="center",
                            ),
                            rx.badge(
                                State.diagram_type,
                                color_scheme="blue",
                                variant="soft",
                                size="3",
                            ),
                        ),
                        spacing="3",
                        width="100%",
                        padding="1.5em",
                        border_radius="12px",
                        bg="white",
                    ),
                ),
                # Detected elements card
                rx.cond(
                    State.detected_elements,
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("grid", size=24, color="#667eea"),
                                rx.heading("Обнаруженные элементы", size="4"),
                                spacing="2",
                                align="center",
                            ),
                            rx.flex(
                                rx.foreach(
                                    State.detected_elements,
                                    lambda el: rx.badge(
                                        el,
                                        color_scheme="green",
                                        variant="soft",
                                        size="2",
                                        padding_x="1em",
                                        padding_y="0.5em",
                                        border_radius="8px",
                                    ),
                                ),
                                wrap="wrap",
                                spacing="2",
                                gap="2",
                            ),
                        ),
                        spacing="3",
                        width="100%",
                        padding="1.5em",
                        border_radius="12px",
                        bg="white",
                    ),
                ),
                # Relationships card
                rx.cond(
                    State.relationships,
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("share-2", size=24, color="#667eea"),
                                rx.heading("Связи между элементами", size="4"),
                                spacing="2",
                                align="center",
                            ),
                            rx.vstack(
                                rx.foreach(
                                    State.relationships,
                                    lambda rel: rx.hstack(
                                        rx.icon("link-2", size=14, color="#764ba2"),
                                        rx.text(
                                            rel, font_size="0.95em", color="#2d3748"
                                        ),
                                        spacing="2",
                                        align="center",
                                    ),
                                ),
                                spacing="2",
                                align="start",
                                width="100%",
                            ),
                        ),
                        spacing="3",
                        width="100%",
                        padding="1.5em",
                        border_radius="12px",
                        bg="white",
                    ),
                ),
                # Step by step description card
                rx.cond(
                    State.step_by_step_description,
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("list", size=24, color="#667eea"),
                                rx.heading("Пошаговое описание", size="4"),
                                spacing="2",
                                align="center",
                            ),
                            rx.divider(),
                            # Current step card
                            rx.card(
                                rx.vstack(
                                    rx.text(
                                        State.step_by_step_description[
                                            State.current_step
                                        ],
                                        font_size="1.1em",
                                        font_weight="bold",
                                        color="#2d3748",
                                    ),
                                    rx.badge(
                                        f"Шаг {State.current_step + 1} из {State.step_by_step_description.length()}",
                                        color_scheme="blue",
                                        variant="soft",
                                        size="2",
                                    ),
                                ),
                                spacing="3",
                                width="100%",
                                bg="#f7fafc",
                                border_radius="12px",
                                padding="1.5em",
                            ),
                            # Navigation buttons
                            rx.hstack(
                                rx.button(
                                    rx.hstack(
                                        rx.icon("arrow-left", size=16),
                                        rx.text("Предыдущий"),
                                    ),
                                    on_click=State.prev_step,
                                    disabled=State.current_step == 0,
                                    variant="outline",
                                    bg="white",
                                ),
                                rx.button(
                                    rx.hstack(
                                        rx.text("Следующий"),
                                        rx.icon("arrow-right", size=16),
                                    ),
                                    on_click=State.next_step,
                                    disabled=State.current_step
                                    >= State.step_by_step_description.length() - 1,
                                    color_scheme="blue",
                                ),
                                justify="center",
                                spacing="4",
                                width="100%",
                            ),
                            # Progress bar
                            rx.progress(
                                value=(State.current_step + 1)
                                / State.step_by_step_description.length()
                                * 100,
                                color_scheme="blue",
                                width="100%",
                            ),
                        ),
                        spacing="4",
                        width="100%",
                        padding="1.5em",
                        border_radius="12px",
                        bg="white",
                    ),
                ),
                spacing="6",
                align="stretch",
            ),
            max_width="1000px",
            padding_x="2em",
            padding_y="3em",
            min_height="calc(100vh - 140px)",
        ),
        footer(),
    )


# Custom styles
style = {
    "font_family": "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "background_attachment": "fixed",
}

# Создаём приложение
app = rx.App(
    style=style,
)

app.add_page(index, route="/", title="DiagramLIT — Анализ диаграмм")
app.add_page(results, route="/results", title="Результаты — DiagramLIT")
