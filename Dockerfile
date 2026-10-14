FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model101.pth .
COPY main.py .
COPY classes.txt .


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]