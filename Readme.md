# WIP libgccjit codegen backend for rust

[![Chat on IRC](https://img.shields.io/badge/irc.libera.chat-%23rustc__codegen__gcc-blue.svg)](https://web.libera.chat/#rustc_codegen_gcc)
[![Chat on Matrix](https://img.shields.io/badge/matrix.org-%23rustc__codegen__gcc-blue.svg)](https://matrix.to/#/#rustc_codegen_gcc:matrix.org)

This is a GCC codegen for rustc, which means it can be loaded by the existing rustc frontend, but benefits from GCC: more architectures are supported and GCC's optimizations are used.

**Despite its name, libgccjit can be used for ahead-of-time compilation, as is used here.**

## Motivation

The primary goal of this project is to be able to compile Rust code on platforms unsupported by LLVM.
A secondary goal is to check if using the gcc backend will provide any run-time speed improvement for the programs compiled using rustc.

## Building

**This requires a patched libgccjit in order to work.
You need to use my [fork of gcc](https://github.com/antoyo/gcc) which already includes these patches.**

```bash
$ cp config.example.toml config.toml
```

If don't need to test GCC patches you wrote in our GCC fork, then the default configuration should
be all you need. You can update the `rustc_codegen_gcc` without worrying about GCC.

### Building with your own GCC version

If you wrote a patch for GCC and want to test it without this backend, you will need
to do a few more things.

To build it (most of these instructions come from [here](https://gcc.gnu.org/onlinedocs/jit/internals/index.html), so don't hesitate to take a look there if you encounter an issue):

```bash
$ git clone https://github.com/antoyo/gcc
$ sudo apt install flex libmpfr-dev libgmp-dev libmpc3 libmpc-dev
$ mkdir gcc-build gcc-install
$ cd gcc-build
$ ../gcc/configure \
    --enable-host-shared \
    --enable-languages=jit \
    --enable-checking=release \ # it enables extra checks which allow to find bugs
    --disable-bootstrap \
    --disable-multilib \
    --prefix=$(pwd)/../gcc-install
$ make -j4 # You can replace `4` with another number depending on how many cores you have.
```

If you want to run libgccjit tests, you will need to also enable the C++ language in the `configure`:

```bash
--enable-languages=jit,c++
```

Then to run libgccjit tests:

```bash
$ cd gcc # from the `gcc-build` folder
$ make check-jit
# To run one specific test:
$ make check-jit RUNTESTFLAGS="-v -v -v jit.exp=jit.dg/test-asm.cc"
```

**Put the path to your custom build of libgccjit in the file `config.toml`.**

You now need to set the `gcc-path` value in `config.toml` with the result of this command:

```bash
$ dirname $(readlink -f `find . -name libgccjit.so`)
```

and to comment the `download-gccjit` setting:

```toml
gcc-path = "[MY PATH]"
# download-gccjit = true
```

Then you can run commands like this:

```bash
$ ./y.sh prepare # download and patch sysroot src and install hyperfine for benchmarking
$ ./y.sh build --sysroot --release
```

To run the tests:

```bash
$ ./y.sh test --release
```

## Usage

`$CG_GCCJIT_DIR` is the directory you cloned this repo into in the following instructions:

```bash
export CG_GCCJIT_DIR=[the full path to rustc_codegen_gcc]
```

### Cargo

```bash
$ CHANNEL="release" $CG_GCCJIT_DIR/y.sh cargo run
```

If you compiled cg_gccjit in debug mode (aka you didn't pass `--release` to `./y.sh test`) you should use `CHANNEL="debug"` instead or omit `CHANNEL="release"` completely.

### LTO

To use LTO, you need to set the variable `FAT_LTO=1` and `EMBED_LTO_BITCODE=1` in addition to setting `lto = "fat"` in the `Cargo.toml`.
Don't set `FAT_LTO` when compiling the sysroot, though: only set `EMBED_LTO_BITCODE=1`.

Failing to set `EMBED_LTO_BITCODE` will give you the following error:

```
error: failed to copy bitcode to object file: No such file or directory (os error 2)
```

### Rustc

If you want to run `rustc` directly, you can do so with:

```bash
$ ./y.sh rustc my_crate.rs
```

You can do the same manually (although we don't recommend it):

```bash
$ LIBRARY_PATH="[gcc-path value]" LD_LIBRARY_PATH="[gcc-path value]" rustc +$(cat $CG_GCCJIT_DIR/rust-toolchain | grep 'channel' | cut -d '=' -f 2 | sed 's/"//g' | sed 's/ //g') -Cpanic=abort -Zcodegen-backend=$CG_GCCJIT_DIR/target/release/librustc_codegen_gcc.so --sysroot $CG_GCCJIT_DIR/build_sysroot/sysroot my_crate.rs
```

## Env vars

<dl>
    <dt>CG_GCCJIT_INCR_CACHE_DISABLED</dt>
    <dd>Don't cache object files in the incremental cache. Useful during development of cg_gccjit
    to make it possible to use incremental mode for all analyses performed by rustc without caching
    object files when their content should have been changed by a change to cg_gccjit.</dd>
    <dt>CG_GCCJIT_DISPLAY_CG_TIME</dt>
    <dd>Display the time it took to perform codegen for a crate</dd>
    <dt>CG_RUSTFLAGS</dt>
    <dd>Send additional flags to rustc. Can be used to build the sysroot without unwinding by setting `CG_RUSTFLAGS=-Cpanic=abort`.</dd>
    <dt>CG_GCCJIT_DUMP_TO_FILE</dt>
    <dd>Dump a C-like representation to /tmp/gccjit_dumps and enable debug info in order to debug this C-like representation.</dd>
</dl>

## Extra documentation

More specific documentation is available in the [`doc`](./doc) folder:

 * [Common errors](./doc/errors.md)
 * [Debugging GCC LTO](./doc/debugging-gcc-lto.md)
 * [Debugging libgccjit](./doc/debugging-libgccjit.md)
 * [Git subtree sync](./doc/subtree.md)
 * [List of useful commands](./doc/tips.md)
 * [Send a patch to GCC](./doc/sending-gcc-patch.md)

## Licensing

While this crate is licensed under a dual Apache/MIT license, it links to `libgccjit` which is under the GPLv3+ and thus, the resulting toolchain (rustc + GCC codegen) will need to be released under the GPL license.

However, programs compiled with `rustc_codegen_gcc` do not need to be released under a GPL license.
