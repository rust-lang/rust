# Rust with AVR support

[![Build Status](https://travis-ci.org/avr-rust/rust.svg)](https://travis-ci.org/avr-rust/rust)

This project adds support for the [AVR](https://en.wikipedia.org/wiki/Atmel_AVR)
microcontroller to Rust.

It uses the [AVR-LLVM backend](https://github.com/avr-llvm/llvm).

## Compiling

This will compile Rust with AVR support. This will not create a
fully-fledged compiler, however - it does not compile any libraries
such as `libcore` or `liblibc`. To do this, the `--target=avr-atmel-none`
flag must be passed to `configure`, which is not fully supported yet.

``` bash
# grab the avr-rust sources
git clone https://github.com/avr-rust/rust.git

# create a directory to place built files in
mkdir build && cd build

# generate makefiles
../rust/configure

# build rust
make
```

**NOTE**: For debugging, it is best to pass the
`--enable-debug --disable-docs --enable-llvm-assertions --enable-debug-assertions`
flags to `configure`. This will help catch bugs that could be otherwise unnoticed.

**NOTE**: This will create a Rust compiler which targets the same architecture
as the computer you are compiling on. In order to get a fully-fledged AVR
compiler (including `libcore` et al), the `--target=avr-atmel-none` flag must
be passed to `configure`, however, this does not work currently (but is being
worked on).

## Building a full cross compiler

**NOTE**: This does not currently work due to a bug.

This process will compile a `rustc` which can target AVR, and also build
`libcore` which can then be used with AVR programs.

This process is identical to compiling Rust as stated before, although different
flags must be passed to `configure`.

``` bash
# grab the avr-rust sources
git clone https://github.com/avr-rust/rust.git

# create a directory to place built files in
mkdir build && cd build

# generate makefiles
../rust/configure --target=avr-atmel-none --disable-jemalloc

# build rust
make
```


## Usage

AVR support is enabled by passing the `--target avr-atmel-none` flag to `rustc`.

Note that the Rust `libcore` library (essentially required for every Rust program),
must be manually compiled for it to be used, as it will not be built for AVR during
compiler compilation (yet). Work is currently being done in order to allow `libcore`
to be automatically compiled for AVR.
