FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils \
  jq \
  bzip2

# emscripten
COPY scripts/emscripten-wasm.sh /scripts/
COPY wasm32-exp/node.sh /usr/local/bin/node
RUN bash /scripts/emscripten-wasm.sh

# cache
COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# env
ENV PATH=/wasm-install/emscripten:/wasm-install/bin:$PATH
ENV EM_CONFIG=/root/.emscripten

ENV TARGETS=wasm32-experimental-emscripten

ENV RUST_CONFIGURE_ARGS --experimental-targets=WebAssembly

ENV SCRIPT python2.7 ../x.py test --target $TARGETS
