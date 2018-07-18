# Work in progress cranelift codegen backend for rust

> ⚠⚠⚠ This doesn't do anything useful yet ⚠⚠⚠

## Building

```bash
$ git clone https://github.com/bjorn3/rustc_codegen_cranelift.git
$ cd rustc_codegen_cranelift
$ git submodule update --init
$ cargo build
```

## Usage

```bash
$ rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cranelift.so my_crate.rs --crate-type lib -Og
```

> You must compile with `-Og`, because checked binops are not yet supported.

## Building libcore

```bash
$ git clone https://github.com/rust-lang/rust.git
$ cd rust
$ git apply ../0001-Disable-stdsimd-in-libcore.patch
$ git apply ../0002-Disable-u128-and-i128-in-libcore.patch
$ cd ../
$ ./build.sh
```

> ⚠⚠⚠ You will get a panic because of unimplemented stuff ⚠⚠⚠

## Not yet supported

* [ ] Checked binops
* [ ] Statics
* [ ] Drop glue

* [ ] Building libraries
* [ ] Other call abi's
* [ ] Unsized types
* [ ] Slice indexing
* [ ] Sub slice
* [ ] Closures
* [ ] Some rvalue's

* [ ] Inline assembly
* [ ] Custom sections

## Known errors

* [ ] cranelift-module api seems to be used wrong, thus causing panic for some consts
* [ ] cranelift-codegen doesn't have encodings for some instructions for types smaller than I32
* [ ] `thread 'main' panicked at 'assertion failed: !value.has_escaping_regions()', librustc/ty/sty.rs:754:9` in cton_sig_from_mono_fn_sig