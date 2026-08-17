ARG IMAGE=ubuntu:26.04
FROM $IMAGE

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gcc \
    gcc-powerpc64le-linux-gnu \
    libc6-dev \
    libc6-dev-ppc64el-cross \
    qemu-system-ppc \
    qemu-user

ENV TOOLCHAIN_PREFIX=powerpc64le-linux-gnu-
ENV CARGO_TARGET_POWERPC64LE_UNKNOWN_LINUX_GNU_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_POWERPC64LE_UNKNOWN_LINUX_GNU_RUNNER=qemu-ppc64le \
    AR_powerpc64le_unknown_linux_gnu="$TOOLCHAIN_PREFIX"ar \
    CC_powerpc64le_unknown_linux_gnu="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/powerpc64le-linux-gnu \
    RUST_TEST_THREADS=1
