FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    make \
    ninja-build \
    file \
    curl \
    ca-certificates \
    python3 \
    git \
    cmake \
    sudo \
    gdb \
    libssl-dev \
    pkg-config \
    xz-utils \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/ohos-sdk.sh /scripts/
RUN sh /scripts/ohos-sdk.sh

COPY scripts/ohos-openssl.sh /scripts/
RUN sh /scripts/ohos-openssl.sh

COPY scripts/ohos/x86_64-unknown-linux-ohos-clang.sh /usr/local/bin/
COPY scripts/ohos/x86_64-unknown-linux-ohos-clang++.sh /usr/local/bin/

# env
ENV X86_64_UNKNOWN_LINUX_OHOS_OPENSSL_DIR=/opt/ohos-openssl/prelude/x86_64

ENV X86_64_UNKNOWN_LINUX_OHOS_OPENSSL_NO_VENDOR=1

ENV TARGETS=x86_64-unknown-linux-ohos

ENV \
    CC_x86_64_unknown_linux_ohos=/usr/local/bin/x86_64-unknown-linux-ohos-clang.sh \
    AR_x86_64_unknown_linux_ohos=/opt/ohos-sdk/native/llvm/bin/llvm-ar \
    CXX_x86_64_unknown_linux_ohos=/usr/local/bin/x86_64-unknown-linux-ohos-clang++.sh

ENV RUST_CONFIGURE_ARGS \
    --enable-profiler \
    --disable-docs \
    --tools=cargo,clippy,rustdocs,rustfmt,rust-analyzer,rust-analyzer-proc-macro-srv,analysis,src,wasm-component-ld \
    --enable-extended \
    --enable-sanitizers

ENV SCRIPT python3 ../x.py dist --host=$TARGETS --target $TARGETS

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh
