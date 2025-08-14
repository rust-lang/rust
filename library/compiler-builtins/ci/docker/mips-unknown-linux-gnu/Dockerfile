ARG IMAGE=ubuntu:25.04
FROM $IMAGE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev ca-certificates \
    gcc-mips-linux-gnu libc6-dev-mips-cross \
    binfmt-support qemu-user-static qemu-system-mips

ENV TOOLCHAIN_PREFIX=mips-linux-gnu-
ENV CARGO_TARGET_MIPS_UNKNOWN_LINUX_GNU_LINKER="$TOOLCHAIN_PREFIX"gcc \
    CARGO_TARGET_MIPS_UNKNOWN_LINUX_GNU_RUNNER=qemu-mips-static \
    AR_mips_unknown_linux_gnu="$TOOLCHAIN_PREFIX"ar \
    CC_mips_unknown_linux_gnu="$TOOLCHAIN_PREFIX"gcc \
    QEMU_LD_PREFIX=/usr/mips-linux-gnu \
    RUST_TEST_THREADS=1
