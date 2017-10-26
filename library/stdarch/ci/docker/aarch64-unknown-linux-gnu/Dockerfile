FROM ubuntu:17.10
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  ca-certificates \
  libc6-dev \
  gcc-aarch64-linux-gnu \
  libc6-dev-arm64-cross \
  qemu-user \
  make \
  file

ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu" \
    OBJDUMP=aarch64-linux-gnu-objdump
