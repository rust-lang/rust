FROM ubuntu:23.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  gcc-multilib \
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
  llvm-16-tools \
  llvm-16-dev \
  libedit-dev \
  libssl-dev \
  pkg-config \
  zlib1g-dev \
  xz-utils \
  nodejs \
  mingw-w64 \
  # libgccjit dependencies
  flex \
  libmpfr-dev \
  libgmp-dev \
  libmpc3 \
  libmpc-dev \
  && rm -rf /var/lib/apt/lists/*

# Note: libgccjit needs to match the default gcc version for the linker to find it.

# Install powershell (universal package) so we can test x.ps1 on Linux
RUN curl -sL "https://github.com/PowerShell/PowerShell/releases/download/v7.3.1/powershell_7.3.1-1.deb_amd64.deb" > powershell.deb && \
    dpkg -i powershell.deb && \
    rm -f powershell.deb

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# We are disabling CI LLVM since this builder is intentionally using a host
# LLVM, rather than the typical src/llvm-project LLVM.
ENV NO_DOWNLOAD_CI_LLVM 1

# This is not the latest LLVM version, so some components required by tests may
# be missing.
ENV IS_NOT_LATEST_LLVM 1

# Using llvm-link-shared due to libffi issues -- see #34486
ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --llvm-root=/usr/lib/llvm-16 \
      --enable-llvm-link-shared \
      --set rust.thin-lto-import-instr-limit=10

COPY host-x86_64/x86_64-gnu-llvm-16/script.sh /tmp/

COPY host-x86_64/dist-x86_64-linux/shared.sh /scripts/
COPY host-x86_64/dist-x86_64-linux/build-gccjit.sh /scripts/

RUN /scripts/build-gccjit.sh /scripts

ENV SCRIPT /tmp/script.sh
