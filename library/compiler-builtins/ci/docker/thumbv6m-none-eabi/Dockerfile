ARG IMAGE=ubuntu:26.04
FROM $IMAGE

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gcc \
    gcc-arm-none-eabi \
    libc6-dev \
    libnewlib-arm-none-eabi
ENV BUILD_ONLY=1
