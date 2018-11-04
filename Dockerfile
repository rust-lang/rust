FROM ubuntu:18.04

USER root

# Install dependencies.
RUN apt-get update && apt-get install -y build-essential cmake curl git python

# Create a regular user.
RUN useradd -m rustacean

# Copy the code over, create an empty directory for builds.
ADD . /code
# RUN cd /code
RUN mkdir /build && cd /build

# symlink Rust build directory to the /build volume
# RUN mkdir /build && ln -sf /build /code/build

# Compile rust.
# RUN /code/x.py build

# Generate Makefile using settings suitable for an experimental compiler
RUN /code/configure \
    --enable-debug \
    --disable-docs \
    --enable-llvm-assertions \
    --enable-debug-assertions \
    --enable-optimize \
    --enable-llvm-release-debuginfo \
    --experimental-targets=AVR

RUN make
RUN make install

# Drop down to the regular user
USER rustacean

VOLUME /code
VOLUME /build
