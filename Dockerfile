# Use the official Python base image with Python 3.10
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt 

# Copy the application files into the container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501
EXPOSE 8502

# Run the Streamlit apps
CMD . venv/bin/activate && \
    streamlit run api_mistral.py --server.port=8501 & \
    streamlit run app_mistral.py --server.port=8502
