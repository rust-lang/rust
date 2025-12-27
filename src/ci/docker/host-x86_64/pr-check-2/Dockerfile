FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  python3-pip \
  python3-pkg-resources \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils \
  libssl-dev \
  pkg-config \
  strace \
  mingw-w64 \
  && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-29/wasi-sdk-29.0-x86_64-linux.tar.gz | \
  tar -xz
ENV WASI_SDK_PATH=/wasi-sdk-29.0-x86_64-linux

ENV RUST_CONFIGURE_ARGS="--set rust.validate-mir-opts=3"

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV SCRIPT \
        ldd --version && \
        python3 ../x.py test --stage 1 std -- "write_at" || \
        strace -ffe write,pwritev2,pwrite64,pwritev python3 ../x.py test --stage 1 std -- "write_at"
