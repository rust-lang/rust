FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev ca-certificates \
    gcc-mipsel-linux-gnu libc6-dev-mipsel-cross \
    binfmt-support qemu-user-static

ENV CARGO_TARGET_MIPSEL_UNKNOWN_LINUX_GNU_LINKER=mipsel-linux-gnu-gcc \
    CARGO_TARGET_MIPSEL_UNKNOWN_LINUX_GNU_RUNNER=qemu-mipsel-static \
    QEMU_LD_PREFIX=/usr/mipsel-linux-gnu \
    RUST_TEST_THREADS=1
