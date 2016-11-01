FROM ubuntu:16.10

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  gcc make libc6-dev git curl ca-certificates
# Below we're cross-compiling musl for i686 using the system compiler on an
# x86_64 system. This is an awkward thing to be doing and so we have to jump
# through a couple hoops to get musl to be happy. In particular:
#
# * We specifically pass -m32 in CFLAGS and override CC when running ./configure,
#   since otherwise the script will fail to find a compiler.
# * We manually unset CROSS_COMPILE when running make; otherwise the makefile
#   will call the non-existent binary 'i686-ar'.
RUN curl https://www.musl-libc.org/releases/musl-1.1.15.tar.gz | \
    tar xzf - && \
    cd musl-1.1.15 && \
    CC=gcc CFLAGS=-m32 ./configure --prefix=/musl-i686 --disable-shared --target=i686 && \
    make CROSS_COMPILE= install -j4 && \
    cd .. && \
    rm -rf musl-1.1.15
ENV PATH=$PATH:/musl-i686/bin:/rust/bin \
    CC_i686_unknown_linux_musl=musl-gcc
