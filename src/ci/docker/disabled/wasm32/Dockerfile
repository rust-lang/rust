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
  xz-utils

# emscripten
COPY scripts/emscripten.sh /scripts/
RUN bash /scripts/emscripten.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/emsdk-portable
ENV PATH=$PATH:/emsdk-portable/clang/e1.38.15_64bit/
ENV PATH=$PATH:/emsdk-portable/emscripten/1.38.15/
ENV PATH=$PATH:/emsdk-portable/node/8.9.1_64bit/bin/
ENV EMSCRIPTEN=/emsdk-portable/emscripten/1.38.15/
ENV BINARYEN_ROOT=/emsdk-portable/clang/e1.38.15_64bit/binaryen/
ENV EM_CONFIG=/emsdk-portable/.emscripten

ENV TARGETS=wasm32-unknown-emscripten
ENV SCRIPT python2.7 ../x.py test --target $TARGETS
