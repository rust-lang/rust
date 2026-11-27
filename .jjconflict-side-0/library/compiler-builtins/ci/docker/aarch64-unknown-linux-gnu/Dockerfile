ARG IMAGE=ubuntu:26.04
FROM $IMAGE

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gcc \
    gcc-aarch64-linux-gnu \
    libc6-dev \
    libc6-dev-arm64-cross \
    m4 \
    make \
    qemu-user

ENV TOOLCHAIN_PREFIX=aarch64-linux-gnu-
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER=qemu-aarch64 \
    AR_aarch64_unknown_linux_gnu="$TOOLCHAIN_PREFIX"ar \
    CC_aarch64_unknown_linux_gnu="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/aarch64-linux-gnu
