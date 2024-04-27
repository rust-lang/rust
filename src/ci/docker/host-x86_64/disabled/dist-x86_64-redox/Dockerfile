FROM ubuntu:16.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/crosstool-ng.sh /scripts/
RUN sh /scripts/crosstool-ng.sh

WORKDIR /tmp
COPY host-x86_64/dist-various-1/install-x86_64-redox.sh /scripts/
RUN sh /scripts/install-x86_64-redox.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV \
    AR_x86_64_unknown_redox=x86_64-unknown-redox-ar \
    CC_x86_64_unknown_redox=x86_64-unknown-redox-gcc \
    CXX_x86_64_unknown_redox=x86_64-unknown-redox-g++

ENV RUST_CONFIGURE_ARGS --enable-extended
ENV SCRIPT python3 ../x.py dist --host='' --target x86_64-unknown-redox
