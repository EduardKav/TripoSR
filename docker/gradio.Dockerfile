FROM triposr-base

COPY tsr ./tsr
COPY examples ./examples
COPY run.py gradio_app.py ./

# Model weights are downloaded from the Hugging Face hub on first run; mount a volume to keep them.
ENV HF_HOME=/cache/huggingface
EXPOSE 7860
CMD ["python", "gradio_app.py", "--listen", "--port", "7860"]
