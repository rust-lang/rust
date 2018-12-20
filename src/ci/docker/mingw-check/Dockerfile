FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils \
  libssl-dev \
  pkg-config \
  mingw-w64

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUN_CHECK_WITH_PARALLEL_QUERIES 1
ENV SCRIPT python2.7 ../x.py check --target=i686-pc-windows-gnu --host=i686-pc-windows-gnu && \
           python2.7 ../x.py build --stage 0 src/tools/build-manifest && \
           python2.7 ../x.py test --stage 0 src/tools/compiletest
