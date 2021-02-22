FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils \
  libssl-dev \
  pkg-config \
  mingw-w64

RUN curl -sL https://nodejs.org/dist/v14.4.0/node-v14.4.0-linux-x64.tar.xz | tar -xJ
ENV PATH="/node-v14.4.0-linux-x64/bin:${PATH}"
# Install es-check
RUN npm install es-check -g

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY host-x86_64/mingw-check/validate-toolstate.sh /scripts/

ENV RUN_CHECK_WITH_PARALLEL_QUERIES 1
ENV SCRIPT /scripts/validate-toolstate.sh
