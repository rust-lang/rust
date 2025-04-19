ARG IMAGE=ubuntu:24.04
FROM $IMAGE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    gcc \
    gcc-mips64-linux-gnuabi64 \
    libc6-dev \
    libc6-dev-mips64-cross \
    qemu-user-static \
    qemu-system-mips

ENV TOOLCHAIN_PREFIX=mips64-linux-gnuabi64-
ENV CARGO_TARGET_MIPS64_UNKNOWN_LINUX_GNUABI64_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_MIPS64_UNKNOWN_LINUX_GNUABI64_RUNNER=qemu-mips64-static \
    AR_mips64_unknown_linux_gnuabi64="$TOOLCHAIN_PREFIX"ar \
    CC_mips64_unknown_linux_gnuabi64="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/mips64-linux-gnuabi64 \
    RUST_TEST_THREADS=1
