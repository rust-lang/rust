FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        gcc libc6-dev ca-certificates \
        gcc-mips-linux-gnu libc6-dev-mips-cross \
        binfmt-support qemu-user-static qemu-system-mips

ENV CARGO_TARGET_MIPS_UNKNOWN_LINUX_GNU_LINKER=mips-linux-gnu-gcc \
    PATH=$PATH:/rust/bin \
    QEMU_LD_PREFIX=/usr/mips-linux-gnu \
    RUST_TEST_THREADS=1
