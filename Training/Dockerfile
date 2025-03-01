# Use official Python image with Java
FROM python:3.9-slim

# Install OpenJDK-17 instead of 11 to match your environment
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYSPARK_PYTHON=/usr/local/bin/python

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt


# Create the run script
COPY scripts/run.sh /scripts/run.sh
RUN chmod +x /scripts/run.sh

# Set entry point
ENTRYPOINT ["/scripts/run.sh"]
