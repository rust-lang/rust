FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-riscv64-linux-gnu libc6-dev-riscv64-cross \
        llvm

ENV CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc \
    CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUNNER="qemu-riscv64 \
      -L /usr/riscv64-linux-gnu \
      -cpu rv64,zk=true,zks=true,zbb=true,zbc=true \
    " \
    OBJDUMP=llvm-objdump
