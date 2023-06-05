FROM ubuntu:20.04

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
  libssl-dev \
  pkg-config \
  xz-utils \
  mingw-w64 \
  && rm -rf /var/lib/apt/lists/*

# Install powershell (universal package) so we can test x.ps1 on Linux
RUN curl -sL "https://github.com/PowerShell/PowerShell/releases/download/v7.3.1/powershell_7.3.1-1.deb_amd64.deb" > powershell.deb && \
    dpkg -i powershell.deb && \
    rm -f powershell.deb

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS \
 --build=x86_64-unknown-linux-gnu \
 --enable-sanitizers \
 --enable-profiler \
 --enable-compiler-docs

COPY host-x86_64/x86_64-gnu/script.sh /tmp/

ENV SCRIPT /tmp/script.sh
