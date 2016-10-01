FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    ca-certificates curl gcc gcc-arm-none-eabi libc6-dev libcurl4-openssl-dev libssh2-1 libnewlib-dev
ENV AR_thumbv7em_none_eabihf=arm-none-eabi-ar \
    CARGO_TARGET_THUMBV7EM_NONE_EABIHF_LINKER=arm-none-eabi-gcc \
    CC_thumbv7em_none_eabihf=arm-none-eabi-gcc \
