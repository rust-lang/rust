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
$ ./y.sh prepare # download and patch sysroot src and install hyperfine for benchmarking
$ LIBRARY_PATH=$(cat gcc_path) LD_LIBRARY_PATH=$(cat gcc_path) ./y.sh build --release
```

To run the tests:

```bash
$ ./test.sh --release
```

## Usage

`$CG_GCCJIT_DIR` is the directory you cloned this repo into in the following instructions:

```bash
export CG_GCCJIT_DIR=[the full path to rustc_codegen_gcc]
```

### Cargo

```bash
$ CHANNEL="release" $CG_GCCJIT_DIR/cargo.sh run
```

If you compiled cg_gccjit in debug mode (aka you didn't pass `--release` to `./test.sh`) you should use `CHANNEL="debug"` instead or omit `CHANNEL="release"` completely.

To use LTO, you need to set the variable `FAT_LTO=1` and `EMBED_LTO_BITCODE=1` in addition to setting `lto = "fat"` in the `Cargo.toml`.
Don't set `FAT_LTO` when compiling the sysroot, though: only set `EMBED_LTO_BITCODE=1`.

### Rustc

> You should prefer using the Cargo method.

```bash
$ LIBRARY_PATH=$(cat gcc_path) LD_LIBRARY_PATH=$(cat gcc_path) rustc +$(cat $CG_GCCJIT_DIR/rust-toolchain | grep 'channel' | cut -d '=' -f 2 | sed 's/"//g' | sed 's/ //g') -Cpanic=abort -Zcodegen-backend=$CG_GCCJIT_DIR/target/release/librustc_codegen_gcc.so --sysroot $CG_GCCJIT_DIR/build_sysroot/sysroot my_crate.rs
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

## Licensing

While this crate is licensed under a dual Apache/MIT license, it links to `libgccjit` which is under the GPLv3+ and thus, the resulting toolchain (rustc + GCC codegen) will need to be released under the GPL license.

However, programs compiled with `rustc_codegen_gcc` do not need to be released under a GPL license.

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

### `failed to build archive` error

When you get this error:

```
error: failed to build archive: failed to open object file: No such file or directory (os error 2)
```

That can be caused by the fact that you try to compile with `lto = "fat"`, but you didn't compile the sysroot with LTO.
(Not sure if that's the reason since I cannot reproduce anymore. Maybe it happened when forgetting setting `FAT_LTO`.)

### ld: cannot find crtbegin.o

When compiling an executable with libgccijt, if setting the `*LIBRARY_PATH` variables to the install directory, you will get the following errors:

```
ld: cannot find crtbegin.o: No such file or directory
ld: cannot find -lgcc: No such file or directory
ld: cannot find -lgcc: No such file or directory
libgccjit.so: error: error invoking gcc driver
```

To fix this, set the variables to `gcc-build/build/gcc`.

### How to debug GCC LTO

Run do the command with `-v -save-temps` and then extract the `lto1` line from the output and run that under the debugger.

### How to send arguments to the GCC linker

```
CG_RUSTFLAGS="-Clink-args=-save-temps -v" ../cargo.sh build
```

### How to see the personality functions in the asm dump

```
CG_RUSTFLAGS="-Clink-arg=-save-temps -v -Clink-arg=-dA" ../cargo.sh build
```

### How to see the LLVM IR for a sysroot crate

```
cargo build -v --target x86_64-unknown-linux-gnu -Zbuild-std
# Take the command from the output and add --emit=llvm-ir
```

### To prevent the linker from unmangling symbols

Run with:

```
COLLECT_NO_DEMANGLE=1
```

### How to use a custom-build rustc

 * Build the stage2 compiler (`rustup toolchain link debug-current build/x86_64-unknown-linux-gnu/stage2`).
 * Clean and rebuild the codegen with `debug-current` in the file `rust-toolchain`.

### How to install a forked git-subtree

Using git-subtree with `rustc` requires a patched git to make it work.
The PR that is needed is [here](https://github.com/gitgitgadget/git/pull/493).
Use the following instructions to install it:

```bash
git clone git@github.com:tqc/git.git
cd git
git checkout tqc/subtree
make
make install
cd contrib/subtree
make
cp git-subtree ~/bin
```

Then, do a sync with this command:

```bash
PATH="$HOME/bin:$PATH" ~/bin/git-subtree push -P compiler/rustc_codegen_gcc/ ../rustc_codegen_gcc/ sync_branch_name
cd ../rustc_codegen_gcc
git checkout master
git pull
git checkout sync_branch_name
git merge master
```

TODO: write a script that does the above.

https://rust-lang.zulipchat.com/#narrow/stream/301329-t-devtools/topic/subtree.20madness/near/258877725

### How to use [mem-trace](https://github.com/antoyo/mem-trace)

`rustc` needs to be built without `jemalloc` so that `mem-trace` can overload `malloc` since `jemalloc` is linked statically, so a `LD_PRELOAD`-ed library won't a chance to intercept the calls to `malloc`.

### How to generate GIMPLE

If you need to check what gccjit is generating (GIMPLE), then take a look at how to
generate it in [gimple.md](./doc/gimple.md).

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
 * (might not be necessary) Disable the compilation of libstd.so (and possibly libcore.so?): Remove dylib from build_sysroot/sysroot_src/library/std/Cargo.toml.
