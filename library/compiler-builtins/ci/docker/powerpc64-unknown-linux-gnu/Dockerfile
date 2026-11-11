ARG IMAGE=ubuntu:26.04
FROM $IMAGE

RUN apt-get update && apt-get install -y --no-install-recommends \
    binfmt-support \
    ca-certificates \
    gcc \
    gcc-powerpc64-linux-gnu \
    libc6-dev \
    libc6-dev-ppc64-cross \
    qemu-system-ppc \
    qemu-user

ENV TOOLCHAIN_PREFIX=powerpc64-linux-gnu-
ENV CARGO_TARGET_POWERPC64_UNKNOWN_LINUX_GNU_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_POWERPC64_UNKNOWN_LINUX_GNU_RUNNER=qemu-ppc64 \
    AR_powerpc64_unknown_linux_gnu="$TOOLCHAIN_PREFIX"ar \
    CC_powerpc64_unknown_linux_gnu="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/powerpc64-linux-gnu
