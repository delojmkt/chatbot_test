FROM python:3.11

WORKDIR /workspace

COPY . .

RUN pip install --upgrade pip
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]
CMD ["main.py"]