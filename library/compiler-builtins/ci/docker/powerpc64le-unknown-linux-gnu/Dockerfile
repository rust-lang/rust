FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user-static ca-certificates \
        gcc-powerpc64le-linux-gnu libc6-dev-ppc64el-cross \
        qemu-system-ppc

ENV CARGO_TARGET_POWERPC64LE_UNKNOWN_LINUX_GNU_LINKER=powerpc64le-linux-gnu-gcc \
    CC_powerpc64le_unknown_linux_gnu=powerpc64le-linux-gnu-gcc \
    PATH=$PATH:/rust/bin \
    QEMU_LD_PREFIX=/usr/powerpc64le-linux-gnu \
    RUST_TEST_THREADS=1
