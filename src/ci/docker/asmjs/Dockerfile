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
  bzip2

COPY scripts/emscripten.sh /scripts/
RUN bash /scripts/emscripten.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/emsdk-portable
ENV PATH=$PATH:/emsdk-portable/upstream/emscripten/
ENV PATH=$PATH:/emsdk-portable/node/12.9.1_64bit/bin/
ENV BINARYEN_ROOT=/emsdk-portable/upstream/
ENV EM_CONFIG=/emsdk-portable/.emscripten

ENV TARGETS=asmjs-unknown-emscripten

ENV SCRIPT python2.7 ../x.py test --target $TARGETS

# This is almost identical to the wasm32-unknown-emscripten target, so
# running with assertions again is not useful
ENV NO_DEBUG_ASSERTIONS=1
ENV NO_LLVM_ASSERTIONS=1
