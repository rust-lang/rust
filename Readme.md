# Work in progress cranelift codegen backend for rust

> ⚠⚠⚠ This doesn't do anything useful yet ⚠⚠⚠

## Building

```bash
$ git clone https://github.com/bjorn3/rustc_codegen_cranelift.git
$ cd rustc_codegen_cranelift
$ rustup override set nightly
$ git submodule update --init
$ cargo build
```

## Usage

```bash
$ rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cranelift.so my_crate.rs --crate-type lib -Og
```

## Building libcore

```bash
$ rustup component add rust-src
$ ./prepare_libcore.sh
$ ./build.sh
```

> This should stop with `error: aborting due to <...> previous errors`
>
> Please submit a bug if not

## Not yet supported

* Checked binops
* Drop glue

* Building libraries
* Other call abi's
* Unsized types
* Slice indexing
* Sub slice

* Inline assembly
* Custom sections

## Known errors

* cranelift-module api seems to be used wrong, thus causing panic for some consts
* cranelift-codegen doesn't have encodings for some instructions for types smaller than I32
