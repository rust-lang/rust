FROM ubuntu:25.10

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  xz-utils \
  clang

ENV VERSION=v34.0.1

RUN curl -L https://github.com/bytecodealliance/wasmtime/releases/download/${VERSION}/wasmtime-${VERSION}-x86_64-linux.tar.xz | tar xJf -
ENV PATH=$PATH:/wasmtime-${VERSION}-x86_64-linux

ENV CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --dir /checkout/target/wasm32-wasip1/release/deps::."
