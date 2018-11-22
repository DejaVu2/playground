FROM python:3.6

ADD ./examples/ppo-agent /agent

# @TODO to be replaced with `pip install pommerman`
ADD . /pommerman
RUN cd /pommerman && pip install . && \
    pip install tensorflow-gpu && \
    cd /baselines && \
    pip install -e .
# end @TODO

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]
