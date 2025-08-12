# We document platform support for minimum glibc 2.17 and kernel 3.2.
# CentOS 7 has headers for kernel 3.10, but that's fine as long as we don't
# actually use newer APIs in rustc or std without a fallback. It's more
# important that we match glibc for ELF symbol versioning.
FROM centos:7

WORKDIR /build

# CentOS 7 EOL is June 30, 2024, but the repos remain in the vault.
RUN sed -i /etc/yum.repos.d/*.repo -e 's!^mirrorlist!#mirrorlist!' \
  -e 's!^#baseurl=http://mirror.centos.org/!baseurl=https://vault.centos.org/!'
RUN sed -i 's/enabled=1/enabled=0/' /etc/yum/pluginconf.d/fastestmirror.conf

RUN yum upgrade -y && \
    yum install -y \
      automake \
      bzip2 \
      file \
      gcc \
      gcc-c++ \
      git \
      glibc-devel \
      glibc-static \
      libedit-devel \
      libstdc++-devel \
      make \
      ncurses-devel \
      openssl-devel \
      patch \
      perl \
      perl-core \
      pkgconfig \
      python3 \
      unzip \
      wget \
      xz \
      zlib-devel \
      && yum clean all

RUN mkdir -p /rustroot/bin

ENV PATH=/rustroot/bin:$PATH
ENV LD_LIBRARY_PATH=/rustroot/lib64:/rustroot/lib32:/rustroot/lib
ENV PKG_CONFIG_PATH=/rustroot/lib/pkgconfig
WORKDIR /tmp
RUN mkdir /home/user
COPY scripts/shared.sh /tmp/

# Need at least GCC 5.1 to compile LLVM
COPY scripts/build-gcc.sh /tmp/
ENV GCC_VERSION=9.5.0
ENV GCC_BUILD_TARGET=aarch64-unknown-linux-gnu
RUN ./build-gcc.sh && yum remove -y gcc gcc-c++

ENV CC=gcc CXX=g++

# LLVM 17 needs cmake 3.20 or higher.
COPY scripts/cmake.sh /tmp/
RUN ./cmake.sh

# Build LLVM+Clang
COPY scripts/build-clang.sh /tmp/
ENV LLVM_BUILD_TARGETS=AArch64
RUN ./build-clang.sh
ENV CC=clang CXX=clang++

# Build zstd to enable `llvm.libzstd`.
COPY scripts/build-zstd.sh /tmp/
RUN ./build-zstd.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PGO_HOST=aarch64-unknown-linux-gnu
ENV HOSTS=aarch64-unknown-linux-gnu

ENV CPATH=/usr/include/aarch64-linux-gnu/:$CPATH

ENV RUST_CONFIGURE_ARGS \
      --build=aarch64-unknown-linux-gnu \
      --enable-full-tools \
      --enable-profiler \
      --enable-sanitizers \
      --enable-compiler-docs \
      --set target.aarch64-unknown-linux-gnu.linker=clang \
      --set target.aarch64-unknown-linux-gnu.ar=/rustroot/bin/llvm-ar \
      --set target.aarch64-unknown-linux-gnu.ranlib=/rustroot/bin/llvm-ranlib \
      --set llvm.link-shared=true \
      --set llvm.thin-lto=true \
      --set llvm.libzstd=true \
      --set llvm.ninja=false \
      --set rust.debug-assertions=false \
      --set rust.jemalloc \
      --set rust.use-lld=true \
      --set rust.lto=thin \
      --set rust.codegen-units=1

ENV SCRIPT python3 ../x.py build --set rust.debug=true opt-dist && \
      ./build/$HOSTS/stage1-tools-bin/opt-dist linux-ci --  python3 ../x.py dist \
      --host $HOSTS --target $HOSTS --include-default-paths build-manifest bootstrap

ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=clang
ENV LIBCURL_NO_PKG_CONFIG 1
ENV DIST_REQUIRE_ALL_TOOLS 1
