FROM ubuntu:25.04
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  g++ \
  ca-certificates \
  libc6-dev \
  gcc-aarch64-linux-gnu \
  g++-aarch64-linux-gnu \
  libc6-dev-arm64-cross \
  qemu-user \
  make \
  file \
  clang-19 \
  lld

ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu" \
    OBJDUMP=aarch64-linux-gnu-objdump \
    STDARCH_TEST_SKIP_FEATURE=tme
