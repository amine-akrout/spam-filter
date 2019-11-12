FROM ubuntu:latest
RUN apt -y update &&\
    apt -y install python3 python3-pip

WORKDIR /usr/src/app
COPY . .

RUN python3 -m pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["app.py"]