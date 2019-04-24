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

# Install `node`
RUN curl https://nodejs.org/dist/v12.0.0/node-v12.0.0-linux-x64.tar.xz | tar xJf -
ENV PATH=$PATH:/node-v12.0.0-linux-x64/bin

COPY docker/wasm32-unknown-unknown/wasm-entrypoint.sh /wasm-entrypoint.sh
ENTRYPOINT ["/wasm-entrypoint.sh"]
