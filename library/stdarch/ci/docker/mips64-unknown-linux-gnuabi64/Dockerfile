FROM ubuntu:25.10

# gcc-mips64-linux-gnuabi64 not available by default in 25.10
RUN echo "deb http://archive.ubuntu.com/ubuntu noble main universe" >> /etc/apt/sources.list.d/noble-copies.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-mips64-linux-gnuabi64 libc6-dev-mips64-cross \
        qemu-system-mips64 qemu-user

# Use Loongson-3A4000, a CPU with MSA support. Qemu does not define a max CPU for MIPS.
ENV CARGO_TARGET_MIPS64_UNKNOWN_LINUX_GNUABI64_LINKER=mips64-linux-gnuabi64-gcc \
    CARGO_TARGET_MIPS64_UNKNOWN_LINUX_GNUABI64_RUNNER="qemu-mips64 -cpu Loongson-3A4000 -L /usr/mips64-linux-gnuabi64" \
    OBJDUMP=mips64-linux-gnuabi64-objdump
