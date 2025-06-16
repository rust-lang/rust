FROM ubuntu:25.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev qemu-user-static ca-certificates \
    gcc-14-loongarch64-linux-gnu libc6-dev-loong64-cross


ENV CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_GNU_LINKER=loongarch64-linux-gnu-gcc-14 \
    CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-loongarch64-static -cpu max -L /usr/loongarch64-linux-gnu" \
    OBJDUMP=loongarch64-linux-gnu-objdump \
    STDARCH_TEST_SKIP_FEATURE=frecipe
