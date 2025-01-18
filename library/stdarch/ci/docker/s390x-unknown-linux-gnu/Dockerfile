FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
        gcc libc6-dev \
        gcc-s390x-linux-gnu libc6-dev-s390x-cross \
        qemu-user \
        make \
        clang \
        file

ENV CARGO_TARGET_S390X_UNKNOWN_LINUX_GNU_LINKER=s390x-linux-gnu-gcc \
    CARGO_TARGET_S390X_UNKNOWN_LINUX_GNU_RUNNER="qemu-s390x -L /usr/s390x-linux-gnu" \
    OBJDUMP=s390x-linux-gnu-objdump
