FROM python:3.10

WORKDIR /code
 
COPY ./requirements.txt /code/requirements.txt
 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#COPY ./start.sh /start.sh
#RUN chmod +x /start.sh

COPY ./app /code/app
#COPY ./model_files /code/app/model_files

#CMD ["/start.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]