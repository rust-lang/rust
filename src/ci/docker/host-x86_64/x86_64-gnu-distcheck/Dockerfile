# Runs `distcheck`, which is a collection of smoke tests:
# 
# - Run `make check` from an unpacked dist tarball to make sure we can at the
#   minimum run check steps from those sources.
# - Check that selected dist components at least have expected directory shape
#   and crate manifests that cargo can generate a lockfile from.
#
# Refer to `src/bootstrap/src/core/build_steps/test.rs` `Distcheck::run` for
# specifics.
#
# FIXME(#136822): dist components are generally under-tested.

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
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
  libssl-dev \
  pkg-config \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# We are disabling CI LLVM since distcheck is an offline build.
ENV NO_DOWNLOAD_CI_LLVM 1

ENV RUST_CONFIGURE_ARGS --build=x86_64-unknown-linux-gnu --set rust.omit-git-hash=false
ENV SCRIPT python3 ../x.py --stage 2 test distcheck
ENV DIST_SRC 1
