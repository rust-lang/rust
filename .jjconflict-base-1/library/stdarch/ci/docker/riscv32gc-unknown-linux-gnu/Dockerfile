FROM ubuntu:25.10

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        wget xz-utils make file llvm

ENV VERSION=2025.07.03

RUN wget "https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/${VERSION}/riscv32-glibc-ubuntu-24.04-gcc-nightly-${VERSION}-nightly.tar.xz" \
    -O riscv-toolchain.tar.xz
RUN tar -xJf riscv-toolchain.tar.xz

ENV CARGO_TARGET_RISCV32GC_UNKNOWN_LINUX_GNU_LINKER=/riscv/bin/riscv32-unknown-linux-gnu-gcc \
    CARGO_TARGET_RISCV32GC_UNKNOWN_LINUX_GNU_RUNNER="qemu-riscv32 -cpu max -L /riscv/sysroot" \
    OBJDUMP=llvm-objdump
