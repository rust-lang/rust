ARG IMAGE=ubuntu:24.04
FROM $IMAGE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc-multilib m4 make libc6-dev ca-certificates
