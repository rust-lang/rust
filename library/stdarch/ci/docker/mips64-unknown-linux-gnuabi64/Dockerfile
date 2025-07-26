FROM ubuntu:25.10

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-mips64-linux-gnuabi64 libc6-dev-mips64-cross \
        qemu-system-mips64 qemu-user

ENV CARGO_TARGET_MIPS64_UNKNOWN_LINUX_GNUABI64_LINKER=mips64-linux-gnuabi64-gcc \
    CARGO_TARGET_MIPS64_UNKNOWN_LINUX_GNUABI64_RUNNER="qemu-mips64 -L /usr/mips64-linux-gnuabi64" \
    OBJDUMP=mips64-linux-gnuabi64-objdump