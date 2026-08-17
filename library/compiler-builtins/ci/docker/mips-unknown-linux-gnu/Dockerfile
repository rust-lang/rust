ARG IMAGE=ubuntu:26.04
FROM $IMAGE

RUN apt-get update && apt-get install -y --no-install-recommends \
    binfmt-support \
    ca-certificates \
    gcc \
    gcc-mips-linux-gnu \
    libc6-dev \
    libc6-dev-mips-cross \
    qemu-system-mips \
    qemu-user

ENV TOOLCHAIN_PREFIX=mips-linux-gnu-
ENV CARGO_TARGET_MIPS_UNKNOWN_LINUX_GNU_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_MIPS_UNKNOWN_LINUX_GNU_RUNNER=qemu-mips \
    AR_mips_unknown_linux_gnu="$TOOLCHAIN_PREFIX"ar \
    CC_mips_unknown_linux_gnu="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/mips-linux-gnu \
    RUST_TEST_THREADS=1
