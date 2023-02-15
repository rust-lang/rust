FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  xz-utils \
  clang

RUN curl -L https://github.com/bytecodealliance/wasmtime/releases/download/dev/wasmtime-dev-$(arch)-linux.tar.xz | tar xJf -
ENV PATH=$PATH:/wasmtime-dev-$(arch)-linux

ENV CARGO_TARGET_WASM32_WASI_RUNNER="wasmtime \
  --wasm-features=threads,relaxed-simd \
  --mapdir .::/checkout/target/wasm32-wasi/release/deps \
  --"
