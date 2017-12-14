FROM ubuntu:17.10

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-powerpc64le-linux-gnu libc6-dev-ppc64el-cross \
        qemu-system-ppc file make

ENV CARGO_TARGET_POWERPC64LE_UNKNOWN_LINUX_GNU_LINKER=powerpc64le-linux-gnu-gcc \
    CARGO_TARGET_POWERPC64LE_UNKNOWN_LINUX_GNU_RUNNER="qemu-ppc64le -L /usr/powerpc64le-linux-gnu" \
    CC=powerpc64le-linux-gnu-gcc \
    OBJDUMP=powerpc64le-linux-gnu-objdump
