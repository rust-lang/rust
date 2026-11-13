// issue-link: https://github.com/rust-lang/rust/issues/160862
// The improper_ctypes lint should fire even when the extern fn comes from a cross-crate macro.

//@ aux-build: cross_crate_macro.rs
//@ check-pass

extern crate cross_crate_macro;

cross_crate_macro::make_extern_fn!();
//~^ WARN `extern` fn uses type `String`, which is not FFI-safe

fn main() {}
