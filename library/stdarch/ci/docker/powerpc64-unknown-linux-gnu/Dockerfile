FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-powerpc64-linux-gnu libc6-dev-ppc64-cross \
        qemu-system-ppc file make

ENV CARGO_TARGET_POWERPC64_UNKNOWN_LINUX_GNU_LINKER=powerpc64-linux-gnu-gcc \
    CARGO_TARGET_POWERPC64_UNKNOWN_LINUX_GNU_RUNNER="qemu-ppc64 -L /usr/powerpc64-linux-gnu" \
    CC=powerpc64-linux-gnu-gcc \
    OBJDUMP=powerpc64-linux-gnu-objdump
