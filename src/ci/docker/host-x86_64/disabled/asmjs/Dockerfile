FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils \
  bzip2

COPY scripts/emscripten.sh /scripts/
RUN bash /scripts/emscripten.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/emsdk-portable
ENV PATH=$PATH:/emsdk-portable/upstream/emscripten/
ENV PATH=$PATH:/emsdk-portable/node/12.9.1_64bit/bin/
ENV BINARYEN_ROOT=/emsdk-portable/upstream/

ENV TARGETS=asmjs-unknown-emscripten

# Use -O1 optimizations in the link step to reduce time spent optimizing JS.
ENV EMCC_CFLAGS=-O1

# Emscripten installation is user-specific
ENV NO_CHANGE_USER=1

ENV SCRIPT python3 ../x.py --stage 2 test --host='' --target $TARGETS

# This is almost identical to the wasm32-unknown-emscripten target, so
# running with assertions again is not useful
ENV NO_DEBUG_ASSERTIONS=1
ENV NO_LLVM_ASSERTIONS=1
ENV NO_OVERFLOW_CHECKS=1
