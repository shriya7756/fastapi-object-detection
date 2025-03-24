# Use the official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn opencv-python-headless

# Expose the FastAPI port
EXPOSE 8000

# Command to run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
