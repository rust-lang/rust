FROM ubuntu:25.10

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates \
        gcc-powerpc64-linux-gnu libc6-dev-ppc64-cross \
        file make

ENV CARGO_TARGET_POWERPC64_UNKNOWN_LINUX_GNU_LINKER=powerpc64-linux-gnu-gcc \
    CARGO_TARGET_POWERPC64_UNKNOWN_LINUX_GNU_RUNNER="qemu-ppc64 -cpu power11 -L /usr/powerpc64-linux-gnu" \
    CC=powerpc64-linux-gnu-gcc \
    OBJDUMP=powerpc64-linux-gnu-objdump \
    STDARCH_TEST_SKIP_FEATURE=vsx \
#   These 2 tests have erratic behaviour with qemu, see https://gitlab.com/qemu-project/qemu/-/issues/1623#note_2449012173
    STDARCH_TEST_SKIP_FUNCTION=vec_lde_u16,vec_lde_u32
