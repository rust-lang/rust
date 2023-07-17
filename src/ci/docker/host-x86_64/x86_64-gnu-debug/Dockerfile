FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  python3-dev \
  libxml2-dev \
  libncurses-dev \
  libedit-dev \
  swig \
  doxygen \
  git \
  cmake \
  sudo \
  gdb \
  libssl-dev \
  pkg-config \
  xz-utils \
  lld \
  clang \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUSTBUILD_FORCE_CLANG_BASED_TESTS 1
ENV RUN_CHECK_WITH_PARALLEL_QUERIES 1

# llvm.use-linker conflicts with downloading CI LLVM
ENV NO_DOWNLOAD_CI_LLVM 1

ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --enable-debug \
      --enable-lld \
      --set llvm.use-linker=lld \
      --set target.x86_64-unknown-linux-gnu.linker=clang \
      --set target.x86_64-unknown-linux-gnu.cc=clang \
      --set target.x86_64-unknown-linux-gnu.cxx=clang++

ENV SCRIPT \
  python3 ../x.py --stage 2 build && \
  python3 ../x.py --stage 2 test tests/run-make-fulldeps --test-args clang
