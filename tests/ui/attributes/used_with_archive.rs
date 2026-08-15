//! Ensure that `#[used]` in archives are correctly registered.
//!
//! Regression test for https://github.com/rust-lang/rust/issues/133491.

//@ run-pass
//@ check-run-results
//@ aux-build: used_pre_main_constructor.rs

//@ ignore-wasm ctor doesn't work on WASM

// Make sure `rustc` links the archive, but intentionally do not import/use any items.
extern crate used_pre_main_constructor as _;

fn main() {
    println!("main");
}
