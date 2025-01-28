# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# You will also find guides on how best to write your Dockerfile

# Use a Python base image (Python 3.12)
FROM python:3.12

# Create a user with UID 1000 (recommended for Docker containers)
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user

# Add the user's local bin directory to the PATH environment variable
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the entire content of your project into the container (application code, model, etc.)
COPY --chown=user . /app

# Start the FastAPI application using Uvicorn, serving it on port 7860 (Hugging Face Spaces defaults to this port)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
