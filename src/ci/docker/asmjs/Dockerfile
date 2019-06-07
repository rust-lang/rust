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

ENV TARGETS=asmjs-unknown-emscripten

ENV RUST_CONFIGURE_ARGS --enable-emscripten --disable-optimize-tests

ENV SCRIPT python2.7 ../x.py test --target $TARGETS \
  src/test/run-pass \
  src/test/run-fail \
  src/libstd \
  src/liballoc \
  src/libcore

# Debug assertions in rustc are largely covered by other builders, and LLVM
# assertions cause this builder to slow down by quite a large amount and don't
# buy us a huge amount over other builders (not sure if we've ever seen an
# asmjs-specific backend assertion trip), so disable assertions for these
# tests.
ENV NO_LLVM_ASSERTIONS=1
ENV NO_DEBUG_ASSERTIONS=1
