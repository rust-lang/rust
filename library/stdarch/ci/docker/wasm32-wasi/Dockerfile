FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  xz-utils \
  clang

RUN curl -L https://github.com/bytecodealliance/wasmtime/releases/download/v0.29.0/wasmtime-v0.29.0-x86_64-linux.tar.xz | tar xJf -
ENV PATH=$PATH:/wasmtime-v0.29.0-x86_64-linux

ENV CARGO_TARGET_WASM32_WASI_RUNNER="wasmtime \
  --enable-simd \
  --enable-threads \
  --mapdir .::/checkout/target/wasm32-wasi/release/deps \
  --"
