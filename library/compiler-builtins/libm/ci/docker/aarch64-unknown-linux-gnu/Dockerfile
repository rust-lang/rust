FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev ca-certificates \
    gcc-aarch64-linux-gnu libc6-dev-arm64-cross \
    qemu-user-static

ENV TOOLCHAIN_PREFIX=aarch64-linux-gnu-
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER=qemu-aarch64-static \
    AR_aarch64_unknown_linux_gnu="$TOOLCHAIN_PREFIX"ar \
    CC_aarch64_unknown_linux_gnu="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/aarch64-linux-gnu \
    RUST_TEST_THREADS=1
