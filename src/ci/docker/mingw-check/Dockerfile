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

ENV SCRIPT python2.7 ../x.py check --target=i686-pc-windows-gnu --host=i686-pc-windows-gnu
