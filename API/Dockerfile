FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3.7 python3.7-dev
RUN apt-get -y install python3-pip
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/* ./
CMD ["python3", "dev.py"]
