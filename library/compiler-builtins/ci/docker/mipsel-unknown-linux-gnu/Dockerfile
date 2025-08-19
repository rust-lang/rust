ARG IMAGE=ubuntu:25.04
FROM $IMAGE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev ca-certificates \
    gcc-mipsel-linux-gnu libc6-dev-mipsel-cross \
    binfmt-support qemu-user-static

ENV TOOLCHAIN_PREFIX=mipsel-linux-gnu-
ENV CARGO_TARGET_MIPSEL_UNKNOWN_LINUX_GNU_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_MIPSEL_UNKNOWN_LINUX_GNU_RUNNER=qemu-mipsel-static \
    AR_mipsel_unknown_linux_gnu="$TOOLCHAIN_PREFIX"ar \
    CC_mipsel_unknown_linux_gnu="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/mipsel-linux-gnu \
    RUST_TEST_THREADS=1
