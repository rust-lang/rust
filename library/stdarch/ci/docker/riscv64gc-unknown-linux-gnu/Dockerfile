FROM ubuntu:25.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-riscv64-linux-gnu libc6-dev-riscv64-cross \
        llvm

ENV CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc \
    CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUNNER="qemu-riscv64 -cpu max -L /usr/riscv64-linux-gnu" \
    OBJDUMP=llvm-objdump
