// issue: <https://github.com/rust-lang/rust/issues/11224>
// Test that an unused `extern crate` declaration does not crash the compiler.
//@ run-pass
//@ aux-build:unused-extern-crate-aux.rs

extern crate unused_extern_crate_aux as unused;

pub fn main() {}
