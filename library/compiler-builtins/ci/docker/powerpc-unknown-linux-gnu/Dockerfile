ARG IMAGE=ubuntu:25.04
FROM $IMAGE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev qemu-user-static ca-certificates \
    gcc-powerpc-linux-gnu libc6-dev-powerpc-cross \
    qemu-system-ppc

ENV TOOLCHAIN_PREFIX=powerpc-linux-gnu-
ENV CARGO_TARGET_POWERPC_UNKNOWN_LINUX_GNU_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_POWERPC_UNKNOWN_LINUX_GNU_RUNNER=qemu-ppc-static \
    AR_powerpc_unknown_linux_gnu="$TOOLCHAIN_PREFIX"ar \
    CC_powerpc_unknown_linux_gnu="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/powerpc-linux-gnu \
    RUST_TEST_THREADS=1
