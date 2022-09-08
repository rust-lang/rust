FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++-multilib \
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
  zlib1g-dev \
  lib32z1-dev \
  xz-utils


COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

RUN mkdir -p /config
RUN echo "[rust]" > /config/nopt-std-config.toml
RUN echo "optimize = false" >> /config/nopt-std-config.toml

ENV RUST_CONFIGURE_ARGS --build=i686-unknown-linux-gnu --disable-optimize-tests
ENV SCRIPT python3 ../x.py test --stage 0 --config /config/nopt-std-config.toml library/std \
  && python3 ../x.py --stage 2 test
