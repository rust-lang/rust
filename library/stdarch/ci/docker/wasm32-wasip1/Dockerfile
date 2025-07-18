FROM ubuntu:25.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  xz-utils \
  clang

RUN curl -L https://github.com/bytecodealliance/wasmtime/releases/download/v18.0.2/wasmtime-v18.0.2-x86_64-linux.tar.xz | tar xJf -
ENV PATH=$PATH:/wasmtime-v18.0.2-x86_64-linux

ENV CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --dir /checkout/target/wasm32-wasip1/release/deps::."
