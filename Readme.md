# WIP Cranelift codegen backend for rust

> ⚠⚠⚠ Certain kinds of FFI don't work yet. ⚠⚠⚠

The goal of this project is to create an alternative codegen backend for the rust compiler based on [Cranelift](https://github.com/bytecodealliance/wasmtime/blob/master/cranelift). This has the potential to improve compilation times in debug mode. If your project doesn't use any of the things listed under "Not yet supported", it should probably work fine. If not please open an issue.

## Building

```bash
$ git clone https://github.com/bjorn3/rustc_codegen_cranelift.git
$ cd rustc_codegen_cranelift
$ ./prepare.sh # download and patch sysroot src and install hyperfine for benchmarking
$ ./test.sh --release
```

## Usage

rustc_codegen_cranelift can be used as a near-drop-in replacement for `cargo build` or `cargo run` for existing projects.

Assuming `$cg_clif_dir` is the directory you cloned this repo into and you followed the instructions (`prepare.sh` and `test.sh`).

### Cargo

In the directory with your project (where you can do the usual `cargo build`), run:

```bash
$ $cg_clif_dir/cargo.sh run
```

This should build and run your project with rustc_codegen_cranelift instead of the usual LLVM backend.

If you compiled cg_clif in debug mode (aka you didn't pass `--release` to `./test.sh`) you should set `CHANNEL="debug"`.

### Rustc

> You should prefer using the Cargo method.

```bash
$ rustc +$(cat $cg_clif_dir/rust-toolchain) -Cpanic=abort -Zcodegen-backend=$cg_clif_dir/target/release/librustc_codegen_cranelift.so --sysroot $cg_clif_dir/build_sysroot/sysroot my_crate.rs
```

### Shell

These are a few functions that allow you to easily run rust code from the shell using cg_clif as jit.

```bash
function jit_naked() {
    echo "$@" | CG_CLIF_JIT=1 rustc -Zcodegen-backend=$cg_clif_dir/target/release/librustc_codegen_cranelift.so --sysroot $cg_clif_dir/build_sysroot/sysroot - -Cprefer-dynamic
}

function jit() {
    jit_naked "fn main() { $@ }"
}

function jit_calc() {
    jit 'println!("0x{:x}", ' $@ ');';
}
```

## Env vars

<dl>
    <dt>CG_CLIF_JIT</dt>
    <dd>Enable JIT mode to immediately run a program instead of writing an executable file.</dd>
    <dt>CG_CLIF_JIT_ARGS</dt>
    <dd>When JIT mode is enable pass these arguments to the program.</dd>
    <dt>CG_CLIF_INCR_CACHE_DISABLED</dt>
    <dd>Don't cache object files in the incremental cache. Useful during development of cg_clif
    to make it possible to use incremental mode for all analyses performed by rustc without caching
    object files when their content should have been changed by a change to cg_clif.</dd>
    <dt>CG_CLIF_DISPLAY_CG_TIME</dt>
    <dd>If "1", display the time it took to perform codegen for a crate</dd>
    <dt>CG_CLIF_FUNCTION_SECTIONS</dt>
    <dd>Use a single section for each function. This will often reduce the executable size at the
        cost of making linking significantly slower.</dd>
</dl>

## Not yet supported

* Good non-rust abi support ([several problems](https://github.com/bjorn3/rustc_codegen_cranelift/issues/10))
* Inline assembly ([no cranelift support](https://github.com/bytecodealliance/wasmtime/issues/1041)
    * On Linux there is support for invoking an external assembler for `global_asm!` and `asm!`.
      `llvm_asm!` will remain unimplemented forever. `asm!` doesn't yet support reg classes. You
      have to specify specific registers instead.
* SIMD ([tracked here](https://github.com/bjorn3/rustc_codegen_cranelift/issues/171), some basic things work)
