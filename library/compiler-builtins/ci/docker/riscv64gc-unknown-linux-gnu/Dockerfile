ARG IMAGE=ubuntu:24.04
FROM $IMAGE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev qemu-user-static ca-certificates \
    gcc-riscv64-linux-gnu libc6-dev-riscv64-cross \
    qemu-system-riscv64

ENV TOOLCHAIN_PREFIX=riscv64-linux-gnu-
ENV CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUNNER=qemu-riscv64-static \
    AR_riscv64gc_unknown_linux_gnu="$TOOLCHAIN_PREFIX"ar \
    CC_riscv64gc_unknown_linux_gnu="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/riscv64-linux-gnu \
    RUST_TEST_THREADS=1
