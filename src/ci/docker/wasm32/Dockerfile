FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
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

# emcc seems to need python to specifically be "python" and not "python3"
RUN ln `which python3` /usr/bin/python

ENV PATH=$PATH:/emsdk-portable
ENV PATH=$PATH:/emsdk-portable/upstream/emscripten/
ENV PATH=$PATH:/emsdk-portable/node/12.9.1_64bit/bin/
ENV BINARYEN_ROOT=/emsdk-portable/upstream/

ENV TARGETS=wasm32-unknown-emscripten

# Use -O1 optimizations in the link step to reduce time spent optimizing.
ENV EMCC_CFLAGS=-O1

# Emscripten installation is user-specific
ENV NO_CHANGE_USER=1

# FIXME: Re-enable these tests once https://github.com/rust-lang/cargo/pull/7476
# is picked up by CI
ENV SCRIPT python3 ../x.py test --target $TARGETS \
    --exclude src/libcore \
    --exclude src/liballoc \
    --exclude src/libproc_macro \
    --exclude src/libstd \
    --exclude src/libterm \
    --exclude src/libtest
