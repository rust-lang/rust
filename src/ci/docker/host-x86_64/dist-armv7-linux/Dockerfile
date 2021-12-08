FROM ubuntu:20.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates
WORKDIR /tmp
RUN curl -f https://curl.se/ca/cacert.pem -o cacert.pem

FROM ubuntu:16.04

# The ca-certificates in ubuntu-16 is too old, so update the certificates
# with something more recent.
COPY --from=0 /tmp/cacert.pem /tmp/cacert.pem
ENV CURL_CA_BUNDLE /tmp/cacert.pem

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/crosstool-ng-1.24.sh /scripts/
RUN sh /scripts/crosstool-ng-1.24.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
USER rustbuild
WORKDIR /tmp

COPY host-x86_64/dist-armv7-linux/build-toolchains.sh host-x86_64/dist-armv7-linux/armv7-linux-gnueabihf.config /tmp/
RUN ./build-toolchains.sh

USER root

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

ENV PATH=$PATH:/x-tools/armv7-unknown-linux-gnueabihf/bin

ENV CC_armv7_unknown_linux_gnueabihf=armv7-unknown-linux-gnueabihf-gcc \
    AR_armv7_unknown_linux_gnueabihf=armv7-unknown-linux-gnueabihf-ar \
    CXX_armv7_unknown_linux_gnueabihf=armv7-unknown-linux-gnueabihf-g++

ENV HOSTS=armv7-unknown-linux-gnueabihf

ENV RUST_CONFIGURE_ARGS --enable-full-tools --disable-docs
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
