# WIP libgccjit codegen backend for rust

[![Chat on IRC](https://img.shields.io/badge/irc.libera.chat-%23rustc__codegen__gcc-blue.svg)](https://web.libera.chat/#rustc_codegen_gcc)

This is a GCC codegen for rustc, which means it can be loaded by the existing rustc frontend, but benefits from GCC: more architectures are supported and GCC's optimizations are used.

**Despite its name, libgccjit can be used for ahead-of-time compilation, as is used here.**

## Motivation

The primary goal of this project is to be able to compile Rust code on platforms unsupported by LLVM.
A secondary goal is to check if using the gcc backend will provide any run-time speed improvement for the programs compiled using rustc.

## Building

**This requires a patched libgccjit in order to work.
The patches in [this repository](https://github.com/antoyo/libgccjit-patches) need to be applied.
(Those patches should work when applied on master, but in case it doesn't work, they are known to work when applied on 079c23cfe079f203d5df83fea8e92a60c7d7e878.)
You can also use my [fork of gcc](https://github.com/antoyo/gcc) which already includes these patches.**

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

**Put the path to your custom build of libgccjit in the file `gcc_path`.**

```bash
$ dirname $(readlink -f `find . -name libgccjit.so`) > gcc_path
```

You also need to set RUST_COMPILER_RT_ROOT:

```bash
$ git clone https://github.com/llvm/llvm-project llvm --depth 1 --single-branch
$ export RUST_COMPILER_RT_ROOT="$PWD/llvm/compiler-rt"
```

Then you can run commands like this:

```bash
$ ./prepare.sh # download and patch sysroot src and install hyperfine for benchmarking
$ LIBRARY_PATH=$(cat gcc_path) LD_LIBRARY_PATH=$(cat gcc_path) ./build.sh --release
```

To run the tests:

```bash
$ ./test.sh --release
```

## Usage

`$cg_gccjit_dir` is the directory you cloned this repo into in the following instructions.

### Cargo

```bash
$ CHANNEL="release" $cg_gccjit_dir/cargo.sh run
```

If you compiled cg_gccjit in debug mode (aka you didn't pass `--release` to `./test.sh`) you should use `CHANNEL="debug"` instead or omit `CHANNEL="release"` completely.

### Rustc

> You should prefer using the Cargo method.

```bash
$ rustc +$(cat $cg_gccjit_dir/rust-toolchain) -Cpanic=abort -Zcodegen-backend=$cg_gccjit_dir/target/release/librustc_codegen_gcc.so --sysroot $cg_gccjit_dir/build_sysroot/sysroot my_crate.rs
```

## Env vars

<dl>
    <dt>CG_GCCJIT_INCR_CACHE_DISABLED</dt>
    <dd>Don't cache object files in the incremental cache. Useful during development of cg_gccjit
    to make it possible to use incremental mode for all analyses performed by rustc without caching
    object files when their content should have been changed by a change to cg_gccjit.</dd>
    <dt>CG_GCCJIT_DISPLAY_CG_TIME</dt>
    <dd>Display the time it took to perform codegen for a crate</dd>
</dl>

## Debugging

Sometimes, libgccjit will crash and output an error like this:

```
during RTL pass: expand
libgccjit.so: error: in expmed_mode_index, at expmed.h:249
0x7f0da2e61a35 expmed_mode_index
	../../../gcc/gcc/expmed.h:249
0x7f0da2e61aa4 expmed_op_cost_ptr
	../../../gcc/gcc/expmed.h:271
0x7f0da2e620dc sdiv_cost_ptr
	../../../gcc/gcc/expmed.h:540
0x7f0da2e62129 sdiv_cost
	../../../gcc/gcc/expmed.h:558
0x7f0da2e73c12 expand_divmod(int, tree_code, machine_mode, rtx_def*, rtx_def*, rtx_def*, int)
	../../../gcc/gcc/expmed.c:4335
0x7f0da2ea1423 expand_expr_real_2(separate_ops*, rtx_def*, machine_mode, expand_modifier)
	../../../gcc/gcc/expr.c:9240
0x7f0da2cd1a1e expand_gimple_stmt_1
	../../../gcc/gcc/cfgexpand.c:3796
0x7f0da2cd1c30 expand_gimple_stmt
	../../../gcc/gcc/cfgexpand.c:3857
0x7f0da2cd90a9 expand_gimple_basic_block
	../../../gcc/gcc/cfgexpand.c:5898
0x7f0da2cdade8 execute
	../../../gcc/gcc/cfgexpand.c:6582
```

To see the code which causes this error, call the following function:

```c
gcc_jit_context_dump_to_file(ctxt, "/tmp/output.c", 1 /* update_locations */)
```

This will create a C-like file and add the locations into the IR pointing to this C file.
Then, rerun the program and it will output the location in the second line:

```
libgccjit.so: /tmp/something.c:61322:0: error: in expmed_mode_index, at expmed.h:249
```

Or add a breakpoint to `add_error` in gdb and print the line number using:

```
p loc->m_line
p loc->m_filename->m_buffer
```

To print a debug representation of a tree:

```c
debug_tree(expr);
```

(defined in print-tree.h)

To print a debug reprensentation of a gimple struct:

```c
debug_gimple_stmt(gimple_struct)
```

To get the `rustc` command to run in `gdb`, add the `--verbose` flag to `cargo build`.

To have the correct file paths in `gdb` instead of `/usr/src/debug/gcc/libstdc++-v3/libsupc++/eh_personality.cc`:

Maybe by calling the following at the beginning of gdb:

```
set substitute-path /usr/src/debug/gcc /path/to/gcc-repo/gcc
```

TODO(antoyo): but that's not what I remember I was doing.

### How to use a custom-build rustc

 * Build the stage2 compiler (`rustup toolchain link debug-current build/x86_64-unknown-linux-gnu/stage2`).
 * Clean and rebuild the codegen with `debug-current` in the file `rust-toolchain`.

### How to install a forked git-subtree

Using git-subtree with `rustc` requires a patched git to make it work.
The PR that is needed is [here](https://github.com/gitgitgadget/git/pull/493).
Use the following instructions to install it:

```
git clone git@github.com:tqc/git.git
cd git
git checkout tqc/subtree
make
make install
cd contrib/subtree
make
cp git-subtree ~/bin
```

### How to use [mem-trace](https://github.com/antoyo/mem-trace)

`rustc` needs to be built without `jemalloc` so that `mem-trace` can overload `malloc` since `jemalloc` is linked statically, so a `LD_PRELOAD`-ed library won't a chance to intercept the calls to `malloc`.

### How to build a cross-compiling libgccjit

#### Building libgccjit

 * Follow these instructions: https://preshing.com/20141119/how-to-build-a-gcc-cross-compiler/ with the following changes:
 * Configure gcc with `../gcc/configure --enable-host-shared --disable-multilib --enable-languages=c,jit,c++ --disable-bootstrap --enable-checking=release --prefix=/opt/m68k-gcc/ --target=m68k-linux --without-headers`.
 * Some shells, like fish, don't define the environment variable `$MACHTYPE`.
 * Add `CFLAGS="-Wno-error=attributes -g -O2"` at the end of the configure command for building glibc (`CFLAGS="-Wno-error=attributes -Wno-error=array-parameter -Wno-error=stringop-overflow -Wno-error=array-bounds -g -O2"` for glibc 2.31, which is useful for Debian).

#### Configuring rustc_codegen_gcc

 * Set `TARGET_TRIPLE="m68k-unknown-linux-gnu"` in config.sh.
 * Since rustc doesn't support this architecture yet, set it back to `TARGET_TRIPLE="mips-unknown-linux-gnu"` (or another target having the same attributes). Alternatively, create a [target specification file](https://book.avr-rust.com/005.1-the-target-specification-json-file.html) (note that the `arch` specified in this file must be supported by the rust compiler).
 * Set `linker='-Clinker=m68k-linux-gcc'`.
 * Set the path to the cross-compiling libgccjit in `gcc_path`.
 * Comment the line: `context.add_command_line_option("-masm=intel");` in src/base.rs.
 * (might not be necessary) Disable the compilation of libstd.so (and possibly libcore.so?).
