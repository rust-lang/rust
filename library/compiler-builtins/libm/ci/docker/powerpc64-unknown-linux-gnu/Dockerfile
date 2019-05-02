FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev ca-certificates \
    gcc-powerpc64-linux-gnu libc6-dev-ppc64-cross \
    binfmt-support qemu-user-static qemu-system-ppc

ENV CARGO_TARGET_POWERPC64_UNKNOWN_LINUX_GNU_LINKER=powerpc64-linux-gnu-gcc \
    CARGO_TARGET_POWERPC64_UNKNOWN_LINUX_GNU_RUNNER=qemu-ppc64-static \
    CC_powerpc64_unknown_linux_gnu=powerpc64-linux-gnu-gcc \
    QEMU_LD_PREFIX=/usr/powerpc64-linux-gnu \
    RUST_TEST_THREADS=1
