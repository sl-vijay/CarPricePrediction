FROM continuumio/anaconda3:2021.05
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN pip install --upgrade pip
RUN pip install -r requirements-cl.txt
CMD python app1.py