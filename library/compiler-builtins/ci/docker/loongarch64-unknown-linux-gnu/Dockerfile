ARG IMAGE=ubuntu:24.04
FROM $IMAGE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev qemu-user-static ca-certificates \
    gcc-14-loongarch64-linux-gnu libc6-dev-loong64-cross

ENV CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_GNU_LINKER=loongarch64-linux-gnu-gcc-14 \
    CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_GNU_RUNNER=qemu-loongarch64-static \
    AR_loongarch64_unknown_linux_gnu=loongarch64-linux-gnu-ar \
    CC_loongarch64_unknown_linux_gnu=loongarch64-linux-gnu-gcc-14 \
    QEMU_LD_PREFIX=/usr/loongarch64-linux-gnu \
    RUST_TEST_THREADS=1
