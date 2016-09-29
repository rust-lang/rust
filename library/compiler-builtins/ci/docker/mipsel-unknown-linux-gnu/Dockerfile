FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        gcc libc6-dev ca-certificates \
        gcc-mipsel-linux-gnu libc6-dev-mipsel-cross \
        binfmt-support qemu-user-static

ENV CARGO_TARGET_MIPSEL_UNKNOWN_LINUX_GNU_LINKER=mipsel-linux-gnu-gcc \
    PATH=$PATH:/rust/bin \
    QEMU_LD_PREFIX=/usr/mipsel-linux-gnu \
    RUST_TEST_THREADS=1
