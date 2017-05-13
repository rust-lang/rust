FROM ubuntu:16.10

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-powerpc-linux-gnu libc6-dev-powerpc-cross \
        qemu-system-ppc

ENV CARGO_TARGET_POWERPC_UNKNOWN_LINUX_GNU_LINKER=powerpc-linux-gnu-gcc \
    PATH=$PATH:/rust/bin
