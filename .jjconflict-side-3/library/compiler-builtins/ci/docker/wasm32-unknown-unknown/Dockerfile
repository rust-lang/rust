ARG IMAGE=ubuntu:26.04
FROM $IMAGE

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    clang \
    gcc \
    libc6-dev

ENV BUILD_ONLY=1
