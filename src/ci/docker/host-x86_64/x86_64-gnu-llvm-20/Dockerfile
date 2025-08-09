FROM ubuntu:25.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  bzip2 \
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
  lldb-20 \
  llvm-20-tools \
  llvm-20-dev \
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

# The official LLDB symbolic links seems to be broken, this is a workaround to fix them.
RUN ln -sf ../../../liblldb.so.1 /usr/lib/llvm-20/lib/python3.13/site-packages/lldb/_lldb.cpython-313-x86_64-linux-gnu.so && \
    ln -sf ../../../../../x86_64-linux-gnu/libLLVM-20.so /usr/lib/llvm-20/lib/python3/dist-packages/lldb/libLLVM-20.*.so.1 && \
    ln -sf ../../../../../x86_64-linux-gnu/libLLVM-20.so /usr/lib/llvm-20/lib/python3/dist-packages/lldb/libLLVM-20.so.1

# This provides hint for LLDB to find the debug server binary.
ENV LLDB_DEBUGSERVER_PATH=/usr/bin/lldb-server-20

# Install powershell (universal package) so we can test x.ps1 on Linux
# FIXME: need a "universal" version that supports libicu74, but for now it still works to ignore that dep.
RUN curl -sL "https://github.com/PowerShell/PowerShell/releases/download/v7.3.1/powershell_7.3.1-1.deb_amd64.deb" > powershell.deb && \
    dpkg --ignore-depends=libicu72 -i powershell.deb && \
    rm -f powershell.deb

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# We are disabling CI LLVM since this builder is intentionally using a host
# LLVM, rather than the typical src/llvm-project LLVM.
ENV NO_DOWNLOAD_CI_LLVM 1
ENV EXTERNAL_LLVM 1

# Using llvm-link-shared due to libffi issues -- see #34486
ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --llvm-root=/usr/lib/llvm-20 \
      --enable-llvm-link-shared \
      --set build.lldb=lldb-20 \
      --set rust.randomize-layout=true \
      --set rust.thin-lto-import-instr-limit=10

COPY scripts/shared.sh /scripts/

COPY scripts/x86_64-gnu-llvm.sh /scripts/
COPY scripts/x86_64-gnu-llvm2.sh /scripts/
COPY scripts/x86_64-gnu-llvm3.sh /scripts/
COPY scripts/stage_2_test_set1.sh /scripts/
COPY scripts/stage_2_test_set2.sh /scripts/

ENV SCRIPT "Must specify DOCKER_SCRIPT for this image"
