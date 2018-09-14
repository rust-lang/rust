FROM ubuntu:18.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
  ca-certificates \
  clang \
  cmake \
  curl \
  git \
  libc6-dev \
  make \
  python \
  xz-utils

# Install `wasm2wat`
RUN git clone --recursive https://github.com/WebAssembly/wabt
RUN make -C wabt -j$(nproc)
ENV PATH=$PATH:/wabt/bin

# Install `wasm-bindgen-test-runner`
RUN curl -L https://github.com/rustwasm/wasm-bindgen/releases/download/0.2.19/wasm-bindgen-0.2.19-x86_64-unknown-linux-musl.tar.gz \
  | tar xzf -
ENV PATH=$PATH:/wasm-bindgen-0.2.19-x86_64-unknown-linux-musl
ENV CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner

# Install `node`
RUN curl https://nodejs.org/dist/v10.8.0/node-v10.8.0-linux-x64.tar.xz | tar xJf -
ENV PATH=$PATH:/node-v10.8.0-linux-x64/bin
