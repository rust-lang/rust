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

ENV RUST_CONFIGURE_ARGS --build=i686-unknown-linux-gnu
# Exclude some tests that are unlikely to be platform specific, to speed up
# this slow job.
ENV SCRIPT python3 ../x.py --stage 2 test \
  --exclude src/bootstrap \
  --exclude src/test/rustdoc-js \
  --exclude src/tools/error_index_generator \
  --exclude src/tools/linkchecker
