# Work in progress cranelift codegen backend for rust

> ⚠⚠⚠ This doesn't do anything useful yet ⚠⚠⚠

## Building

```bash
$ git clone https://github.com/bjorn3/rustc_codegen_cranelift
$ cd rustc_codegen_cranelift
$ git submodule update --init
$ cargo build
```

## Usage

```bash
$ rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cranelift.so my_crate.rs --crate-type lib -Og
```

> You must compile with `-Og`, because checked binops are not yet supported.

## Not yet supported

* [ ] Checked binops
* [ ] Statics
* [ ] Drop glue
* [ ] Ints cast

* [ ] Building libraries
* [ ] Other call abi's
* [ ] Unsized types
* [ ] Slice indexing
* [ ] Sub slice
* [ ] Closures
* [ ] Some rvalue's

* [ ] Inline assembly
* [ ] Custom sections