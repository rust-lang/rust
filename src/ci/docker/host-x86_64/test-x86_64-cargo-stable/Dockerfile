FROM ubuntu:26.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  libssl-dev \
  pkg-config \
  zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY /scripts/cargo-stable-tests.sh /tmp/cargo-stable-tests.sh
ENV SCRIPT="bash /tmp/cargo-stable-tests.sh"
