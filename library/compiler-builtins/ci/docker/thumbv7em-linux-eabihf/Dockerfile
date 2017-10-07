FROM ubuntu:16.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates curl gcc gcc-arm-none-eabi libc6-dev libcurl4-openssl-dev libssh2-1 libnewlib-dev qemu-user-static
RUN curl -LSfs https://japaric.github.io/trust/install.sh | \
    sh -s -- --git japaric/xargo --tag v0.3.1 --target x86_64-unknown-linux-gnu --to /usr/bin
ENV AR_thumbv7em_linux_eabihf=arm-none-eabi-ar \
    CARGO_TARGET_THUMBV7EM_LINUX_EABIHF_LINKER=arm-none-eabi-gcc \
    CARGO_TARGET_THUMBV7EM_LINUX_EABIHF_RUNNER=qemu-arm-static \
    CC_thumbv7em_linux_eabihf=arm-none-eabi-gcc \
