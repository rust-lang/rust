FROM ubuntu:16.10

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  gcc make libc6-dev git curl ca-certificates
RUN curl https://www.musl-libc.org/releases/musl-1.1.15.tar.gz | \
    tar xzf - && \
    cd musl-1.1.15 && \
    ./configure --prefix=/musl-x86_64 && \
    make install -j4 && \
    cd .. && \
    rm -rf musl-1.1.15
ENV PATH=$PATH:/musl-x86_64/bin:/rust/bin
