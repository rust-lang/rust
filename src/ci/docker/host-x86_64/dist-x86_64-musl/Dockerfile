FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  wget \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  bzip2 \
  xz-utils \
  sudo \
  gdb \
  patch \
  libssl-dev \
  pkg-config \
  zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /build/

COPY scripts/musl-toolchain.sh /build/
# We need to mitigate rust-lang/rust#34978 when compiling musl itself as well
RUN CFLAGS="-Wa,-mrelax-relocations=no -Wa,--compress-debug-sections=none -Wl,--compress-debug-sections=none" \
    CXXFLAGS="-Wa,-mrelax-relocations=no -Wa,--compress-debug-sections=none -Wl,--compress-debug-sections=none" \
    REPLACE_CC=1 bash musl-toolchain.sh x86_64 && rm -rf build

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV HOSTS=x86_64-unknown-linux-musl

ENV RUST_CONFIGURE_ARGS \
      --musl-root-x86_64=/usr/local/x86_64-linux-musl \
      --enable-extended \
      --enable-sanitizers \
      --enable-profiler \
      --enable-lld \
      --set target.x86_64-unknown-linux-musl.crt-static=false \
      --build $HOSTS

# Newer binutils broke things on some vms/distros (i.e., linking against
# unknown relocs disabled by the following flag), so we need to go out of our
# way to produce "super compatible" binaries.
#
# See: https://github.com/rust-lang/rust/issues/34978
# And: https://github.com/rust-lang/rust/issues/59411
ENV CFLAGS_x86_64_unknown_linux_musl="-Wa,-mrelax-relocations=no -Wa,--compress-debug-sections=none \
    -Wl,--compress-debug-sections=none"

# To run native tests replace `dist` below with `test`
ENV SCRIPT python3 ../x.py dist --build $HOSTS
