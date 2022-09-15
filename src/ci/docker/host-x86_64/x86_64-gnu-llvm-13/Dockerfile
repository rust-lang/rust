FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# NOTE: intentionally installs both python2 and python3 so we can test support for both.
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  gcc-multilib \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python2.7 \
  python3.9 \
  git \
  cmake \
  sudo \
  gdb \
  llvm-13-tools \
  llvm-13-dev \
  libedit-dev \
  libssl-dev \
  pkg-config \
  zlib1g-dev \
  xz-utils \
  nodejs

# Install powershell so we can test x.ps1 on Linux
RUN apt-get update && \
    apt-get install -y apt-transport-https software-properties-common && \
    curl -s "https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb" > packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y powershell

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# We are disabling CI LLVM since this builder is intentionally using a host
# LLVM, rather than the typical src/llvm-project LLVM.
ENV NO_DOWNLOAD_CI_LLVM 1
ENV EXTERNAL_LLVM 1

# Using llvm-link-shared due to libffi issues -- see #34486
ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --llvm-root=/usr/lib/llvm-13 \
      --enable-llvm-link-shared \
      --set rust.thin-lto-import-instr-limit=10

# NOTE: intentionally uses all of `x.py`, `x`, and `x.ps1` to make sure they all work on Linux.
ENV SCRIPT ../x.py --stage 2 test --exclude src/tools/tidy && \
           # Run the `mir-opt` tests again but this time for a 32-bit target.
           # This enforces that tests using `// EMIT_MIR_FOR_EACH_BIT_WIDTH` have
           # both 32-bit and 64-bit outputs updated by the PR author, before
           # the PR is approved and tested for merging.
           # It will also detect tests lacking `// EMIT_MIR_FOR_EACH_BIT_WIDTH`,
           # despite having different output on 32-bit vs 64-bit targets.
           ../x --stage 2 test src/test/mir-opt \
                             --host='' --target=i686-unknown-linux-gnu && \
           # Run the UI test suite again, but in `--pass=check` mode
           #
           # This is intended to make sure that both `--pass=check` continues to
           # work.
           #
           ../x.ps1 --stage 2 test src/test/ui --pass=check \
                             --host='' --target=i686-unknown-linux-gnu && \
           # Run tidy at the very end, after all the other tests.
           python2.7 ../x.py --stage 2 test src/tools/tidy
