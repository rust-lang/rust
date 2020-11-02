# WIP Cranelift codegen backend for rust

> ⚠⚠⚠ Certain kinds of FFI don't work yet. ⚠⚠⚠

The goal of this project is to create an alternative codegen backend for the rust compiler based on [Cranelift](https://github.com/bytecodealliance/wasmtime/blob/master/cranelift).
This has the potential to improve compilation times in debug mode.
If your project doesn't use any of the things listed under "Not yet supported", it should work fine.
If not please open an issue.

## Building and testing

```bash
$ git clone https://github.com/bjorn3/rustc_codegen_cranelift.git
$ cd rustc_codegen_cranelift
$ ./prepare.sh # download and patch sysroot src and install hyperfine for benchmarking
$ ./build.sh
```

To run the test suite replace the last command with:

```bash
$ ./test.sh
```

This will implicitly build cg_clif too. Both `build.sh` and `test.sh` accept a `--debug` argument to
build in debug mode.

Alternatively you can download a pre built version from [GHA]. It is listed in the artifacts section
of workflow runs. Unfortunately due to GHA restrictions you need to be logged in to access it.

[GHA]: https://github.com/bjorn3/rustc_codegen_cranelift/actions?query=branch%3Amaster+event%3Apush+is%3Asuccess

## Usage

rustc_codegen_cranelift can be used as a near-drop-in replacement for `cargo build` or `cargo run` for existing projects.

Assuming `$cg_clif_dir` is the directory you cloned this repo into and you followed the instructions (`prepare.sh` and `build.sh` or `test.sh`).

### Cargo

In the directory with your project (where you can do the usual `cargo build`), run:

```bash
$ $cg_clif_dir/build/cargo.sh run
```

This should build and run your project with rustc_codegen_cranelift instead of the usual LLVM backend.

### Rustc

> You should prefer using the Cargo method.

```bash
$ $cg_clif_dir/build/cg_clif my_crate.rs
```

### Jit mode

In jit mode cg_clif will immediately execute your code without creating an executable file.

> This requires all dependencies to be available as dynamic library.
> The jit mode will probably need cargo integration to make this possible.

```bash
$ $cg_clif_dir/build/cargo.sh jit
```

or

```bash
$ $cg_clif_dir/build/cg_clif --jit my_crate.rs
```

### Shell

These are a few functions that allow you to easily run rust code from the shell using cg_clif as jit.

```bash
function jit_naked() {
    echo "$@" | $cg_clif_dir/build/cg_clif - --jit
}

function jit() {
    jit_naked "fn main() { $@ }"
}

function jit_calc() {
    jit 'println!("0x{:x}", ' $@ ');';
}
```

## Env vars

[see env_vars.md](docs/env_vars.md)

## Not yet supported

* Good non-rust abi support ([several problems](https://github.com/bjorn3/rustc_codegen_cranelift/issues/10))
* Inline assembly ([no cranelift support](https://github.com/bytecodealliance/wasmtime/issues/1041)
    * On Linux there is support for invoking an external assembler for `global_asm!` and `asm!`.
      `llvm_asm!` will remain unimplemented forever. `asm!` doesn't yet support reg classes. You
      have to specify specific registers instead.
* SIMD ([tracked here](https://github.com/bjorn3/rustc_codegen_cranelift/issues/171), some basic things work)
