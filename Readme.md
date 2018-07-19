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

> You must compile with `-Og`, because checked binops are not yet supported.

## Building libcore

```bash
$ rustup component add rust-src
$ ./prepare_libcore.sh
$ ./build.sh
```

> ⚠⚠⚠ You will get a panic because of unimplemented stuff ⚠⚠⚠

## Not yet supported

* Checked binops
* Statics
* Drop glue

* Building libraries
* Other call abi's
* Unsized types
* Slice indexing
* Sub slice
* Closures
* Some rvalue's

* Inline assembly
* Custom sections

## Known errors

* cranelift-module api seems to be used wrong, thus causing panic for some consts
* cranelift-codegen doesn't have encodings for some instructions for types smaller than I32

```
[...]
warning: DefId(0/0:128 ~ lib[8787]::f64[0]::{{impl}}[0]::classify[0]):
fn f64::<impl at ./target/libcore/src/libcore/num/f64.rs:156:1: 490:2>::classify(_1: f64) -> num::FpCategory{
[...]
}
warning: stmt _3 = _1
warning: stmt _5 = BitAnd(move _6, const Unevaluated(DefId(0/0:130 ~ lib[8787]::f64[0]::{{impl}}[0]::classify[0]::MAN_MASK[0]), []):u64)
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', libcore/option.rs:345:21
stack backtrace:
[...]
  10: <core::option::Option<T>>::unwrap
             at /checkout/src/libcore/macros.rs:20
  11: rustc_codegen_cranelift::constant::trans_constant
             at src/constant.rs:26
[...]
```
