FROM ubuntu:16.10

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-mips-linux-gnu libc6-dev-mips-cross \
        qemu-system-mips

ENV CARGO_TARGET_MIPS_UNKNOWN_LINUX_GNU_LINKER=mips-linux-gnu-gcc \
    PATH=$PATH:/rust/bin
