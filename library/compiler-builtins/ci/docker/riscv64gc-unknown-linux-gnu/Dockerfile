ARG IMAGE=ubuntu:26.04
FROM $IMAGE

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gcc \
    gcc-riscv64-linux-gnu \
    libc6-dev \
    libc6-dev-riscv64-cross \
    qemu-system-riscv \
    qemu-user

ENV TOOLCHAIN_PREFIX=riscv64-linux-gnu-
ENV CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUNNER=qemu-riscv64 \
    AR_riscv64gc_unknown_linux_gnu="$TOOLCHAIN_PREFIX"ar \
    CC_riscv64gc_unknown_linux_gnu="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/riscv64-linux-gnu \
    RUST_TEST_THREADS=1
