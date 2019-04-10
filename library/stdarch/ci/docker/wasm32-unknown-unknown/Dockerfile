FROM ubuntu:18.04 as node

RUN apt-get update -y && apt-get install -y \
  g++ \
  git \
  make \
  python


# Install `node`
RUN git clone https://github.com/nodejs/node --depth 1
WORKDIR node
RUN ./configure --prefix=/node-install
RUN make -j$(nproc) install

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

COPY --from=node /node-install /
ENV PATH=$PATH:/node-install/bin

COPY docker/wasm32-unknown-unknown/wasm-entrypoint.sh /wasm-entrypoint.sh
ENTRYPOINT ["/wasm-entrypoint.sh"]
