FROM ubuntu:22.04

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
  llvm-14-tools \
  llvm-14-dev \
  libedit-dev \
  libssl-dev \
  pkg-config \
  zlib1g-dev \
  xz-utils \
  nodejs \
  mingw-w64 \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# We are disabling CI LLVM since this builder is intentionally using a host
# LLVM, rather than the typical src/llvm-project LLVM.
ENV NO_DOWNLOAD_CI_LLVM 1

# Using llvm-link-shared due to libffi issues -- see #34486
ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --llvm-root=/usr/lib/llvm-14 \
      --enable-llvm-link-shared \
      --set rust.thin-lto-import-instr-limit=10

ENV SCRIPT ../x.py --stage 1 test --exclude src/tools/tidy && \
           # Run the `mir-opt` tests again but this time for a 32-bit target.
           # This enforces that tests using `// EMIT_MIR_FOR_EACH_BIT_WIDTH` have
           # both 32-bit and 64-bit outputs updated by the PR author, before
           # the PR is approved and tested for merging.
           # It will also detect tests lacking `// EMIT_MIR_FOR_EACH_BIT_WIDTH`,
           # despite having different output on 32-bit vs 64-bit targets.
           ../x.py --stage 1 test tests/mir-opt \
                             --host='' --target=i686-unknown-linux-gnu
