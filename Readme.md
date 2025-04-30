# WIP libgccjit codegen backend for rust

[![Chat on IRC](https://img.shields.io/badge/irc.libera.chat-%23rustc__codegen__gcc-blue.svg)](https://web.libera.chat/#rustc_codegen_gcc)
[![Chat on Matrix](https://img.shields.io/badge/matrix.org-%23rustc__codegen__gcc-blue.svg)](https://matrix.to/#/#rustc_codegen_gcc:matrix.org)

This is a GCC codegen for rustc, which means it can be loaded by the existing rustc frontend, but benefits from GCC: more architectures are supported and GCC's optimizations are used.

**Despite its name, libgccjit can be used for ahead-of-time compilation, as is used here.**

## Motivation

The primary goal of this project is to be able to compile Rust code on platforms unsupported by LLVM.
A secondary goal is to check if using the gcc backend will provide any run-time speed improvement for the programs compiled using rustc.

## Getting Started

Note: **This requires a patched libgccjit in order to work.
You need to use my [fork of gcc](https://github.com/rust-lang/gcc) which already includes these patches.**
The default configuration (see below in the [Quick start](#quick-start) section) will download a `libgccjit` built in the CI that already contains these patches, so you don't need to build this fork yourself if you use the default configuration.

### Dependencies

- rustup: follow instructions on the [official website](https://rustup.rs)
- consider to install DejaGnu which is necessary for running the libgccjit test suite. [website](https://www.gnu.org/software/dejagnu/#downloading)
- additional packages: `flex`, `libmpfr-dev`, `libgmp-dev`, `libmpc3`, `libmpc-dev`
  
### Quick start

1. Clone and configure the repository:
   ```bash
   git clone https://github.com/rust-lang/rustc_codegen_gcc
   cd rustc_codegen_gcc
   cp config.example.toml config.toml
   ```

2. Build and test:
   ```bash
   ./y.sh prepare  # downloads and patches sysroot
   ./y.sh build --sysroot --release
   
   # Verify setup with a simple test
   ./y.sh cargo build --manifest-path tests/hello-world/Cargo.toml
   
   # Run full test suite (expect ~100 failing UI tests)
   ./y.sh test --release
   ```

If don't need to test GCC patches you wrote in our GCC fork, then the default configuration should
be all you need. You can update the `rustc_codegen_gcc` without worrying about GCC.

### Building with your own GCC version

If you wrote a patch for GCC and want to test it without this backend, you will need
to do a few more things.

To build it (most of these instructions come from [here](https://gcc.gnu.org/onlinedocs/jit/internals/index.html), so don't hesitate to take a look there if you encounter an issue):

```bash
$ git clone https://github.com/rust-lang/gcc
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

You have to run these commands, in the corresponding order:

```bash
$ ./y.sh prepare
$ ./y.sh build --sysroot
```
To check if all is  working correctly, run:

 ```bash
$ ./y.sh cargo build --manifest-path tests/hello-world/Cargo.toml
```

### Cargo

```bash
$ CHANNEL="release" $CG_GCCJIT_DIR/y.sh cargo run
```

If you compiled cg_gccjit in debug mode (aka you didn't pass `--release` to `./y.sh test`) you should use `CHANNEL="debug"` instead or omit `CHANNEL="release"` completely.

### LTO

To use LTO, you need to set the variable `EMBED_LTO_BITCODE=1` in addition to setting `lto = "fat"` in the `Cargo.toml`.

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

## Environment variables

 * _**CG_GCCJIT_DUMP_ALL_MODULES**_: Enables dumping of all compilation modules. When set to "1", a dump is created for each module during compilation and stored in `/tmp/reproducers/`.
 * _**CG_GCCJIT_DUMP_MODULE**_: Enables dumping of a specific module. When set with the module name, e.g., `CG_GCCJIT_DUMP_MODULE=module_name`, a dump of that specific module is created in `/tmp/reproducers/`.
 * _**CG_RUSTFLAGS**_: Send additional flags to rustc. Can be used to build the sysroot without unwinding by setting `CG_RUSTFLAGS=-Cpanic=abort`.
 * _**CG_GCCJIT_DUMP_TO_FILE**_: Dump a C-like representation to /tmp/gccjit_dumps and enable debug info in order to debug this C-like representation.
 * _**CG_GCCJIT_DUMP_RTL**_: Dumps RTL (Register Transfer Language) for virtual registers.
 * _**CG_GCCJIT_DUMP_RTL_ALL**_: Dumps all RTL passes.
 * _**CG_GCCJIT_DUMP_TREE_ALL**_: Dumps all tree (GIMPLE) passes.
 * _**CG_GCCJIT_DUMP_IPA_ALL**_: Dumps all Interprocedural Analysis (IPA) passes.
 * _**CG_GCCJIT_DUMP_CODE**_: Dumps the final generated code.
 * _**CG_GCCJIT_DUMP_GIMPLE**_: Dumps the initial GIMPLE representation.
 * _**CG_GCCJIT_DUMP_EVERYTHING**_: Enables dumping of all intermediate representations and passes.
 * _**CG_GCCJIT_KEEP_INTERMEDIATES**_: Keeps intermediate files generated during the compilation process.
 * _**CG_GCCJIT_VERBOSE**_: Enables verbose output from the GCC driver.

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
