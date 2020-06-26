FROM centos:6

WORKDIR /build

# Centos 6 is near EOL and will no longer be available from the usual mirrors,
# so we'll need to switch to http://vault.centos.org/
# RUN sed -i 's/enabled=1/enabled=0/' /etc/yum/pluginconf.d/fastestmirror.conf
# RUN sed -i 's/mirrorlist/#mirrorlist/' /etc/yum.repos.d/*.repo
# RUN sed -i 's|#\(baseurl.*\)mirror.centos.org/centos/$releasever|\1vault.centos.org/6.10|' /etc/yum.repos.d/*.repo

RUN yum install -y epel-release
RUN yum upgrade -y && yum install -y \
      autoconf \
      bzip2 \
      curl \
      file \
      gcc \
      gcc-c++ \
      gettext \
      git \
      glibc-devel.i686 \
      glibc-devel.x86_64 \
      libstdc++-devel.i686 \
      libstdc++-devel.x86_64 \
      make \
      openssl-devel.i686 \
      openssl-devel.x86_64 \
      perl \
      pkgconfig \
      python34 \
      wget \
      which \
      xz \
      zlib-devel.i686 \
      zlib-devel.x86_64

ENV PATH=/rustroot/bin:$PATH
ENV LD_LIBRARY_PATH=/rustroot/lib64:/rustroot/lib
ENV PKG_CONFIG_PATH=/rustroot/lib/pkgconfig
WORKDIR /tmp
COPY dist-x86_64-linux/shared.sh /tmp/

# binutils < 2.22 has a bug where the 32-bit executables it generates
# immediately segfault in Rust, so we need to install our own binutils.
#
# See https://github.com/rust-lang/rust/issues/20440 for more info
COPY dist-x86_64-linux/build-binutils.sh /tmp/
RUN ./build-binutils.sh

# Need a newer version of gcc than centos has to compile LLVM nowadays
COPY dist-x86_64-linux/build-gcc.sh /tmp/
RUN ./build-gcc.sh

# LLVM needs cmake 3.4.3 or higher, and is planning to raise to 3.13.4.
# CentOS 6 only has cmake-2.8.12, or cmake3-3.6.1 in EPEL, so build our own.
COPY dist-x86_64-linux/build-cmake.sh /tmp/
RUN ./build-cmake.sh

# Now build LLVM+Clang, afterwards configuring further compilations to use the
# clang/clang++ compilers.
COPY dist-x86_64-linux/build-clang.sh /tmp/
RUN ./build-clang.sh
ENV CC=clang CXX=clang++

# for sanitizers, we need kernel headers files newer than the ones CentOS ships
# with so we install newer ones here
COPY dist-x86_64-linux/build-headers.sh /tmp/
RUN ./build-headers.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV HOSTS=x86_64-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS \
      --enable-full-tools \
      --enable-sanitizers \
      --enable-profiler \
      --enable-compiler-docs \
      --set target.x86_64-unknown-linux-gnu.linker=clang \
      --set target.x86_64-unknown-linux-gnu.ar=/rustroot/bin/llvm-ar \
      --set target.x86_64-unknown-linux-gnu.ranlib=/rustroot/bin/llvm-ranlib \
      --set llvm.thin-lto=true \
      --set rust.jemalloc
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=clang

# This is the only builder which will create source tarballs
ENV DIST_SRC 1

# When we build cargo in this container, we don't want it to use the system
# libcurl, instead it should compile its own.
ENV LIBCURL_NO_PKG_CONFIG 1

ENV DIST_REQUIRE_ALL_TOOLS 1
