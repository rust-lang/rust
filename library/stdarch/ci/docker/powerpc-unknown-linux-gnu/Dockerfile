FROM ubuntu:25.10

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-powerpc-linux-gnu libc6-dev-powerpc-cross \
        qemu-system-ppc make file

ENV CARGO_TARGET_POWERPC_UNKNOWN_LINUX_GNU_LINKER=powerpc-linux-gnu-gcc \
    CARGO_TARGET_POWERPC_UNKNOWN_LINUX_GNU_RUNNER="qemu-ppc -cpu mpc8610 -L /usr/powerpc-linux-gnu" \
    CC=powerpc-linux-gnu-gcc \
    OBJDUMP=powerpc-linux-gnu-objdump \
    STDARCH_TEST_SKIP_FEATURE=vsx
