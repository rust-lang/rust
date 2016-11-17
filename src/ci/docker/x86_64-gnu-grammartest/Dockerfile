FROM ubuntu:16.10

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python \
  git \
  cmake \
  ccache \
  sudo \
  gdb \
  llvm-3.9-tools \
  libedit-dev \
  zlib1g-dev \
  antlr4 \
  bison \
  flex \
  openjdk-8-jdk-headless

RUN cd /usr/local/lib && curl http://www.antlr.org/download/antlr-4.5.3-complete.jar -o antlr-complete.jar

ENV RUST_CONFIGURE_ARGS \
         --build=x86_64-unknown-linux-gnu \
         --llvm-root=/usr/lib/llvm-3.9
ENV RUST_CHECK_TARGET check-grammar
RUN mkdir /tmp/obj
RUN chmod 777 /tmp/obj
