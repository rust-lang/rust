//! Don't lint on binary crate with non-snake-case names.
//!
//! See <https://github.com/rust-lang/rust/issues/45127>.

//@ revisions: bin_ cdylib_ dylib_ lib_ proc_macro_ rlib_ staticlib_

// Should not fire on binary crates.
//@[bin_] compile-flags: --crate-type=bin
//@[bin_] check-pass

// But should fire on non-binary crates.

// FIXME(#132309): dylib crate type is not supported on wasm; we need a proper
// supports-crate-type directive. Also, needs-dynamic-linking should rule out
// musl since it supports neither dylibs nor cdylibs.
//@[dylib_] ignore-wasm
//@[dylib_] ignore-musl
//@[cdylib_] ignore-musl

//@[dylib_] needs-dynamic-linking
//@[cdylib_] needs-dynamic-linking
//@[proc_macro_] force-host
//@[proc_macro_] no-prefer-dynamic

//@[cdylib_] compile-flags: --crate-type=cdylib
//@[dylib_] compile-flags: --crate-type=dylib
//@[lib_] compile-flags: --crate-type=lib
//@[proc_macro_] compile-flags: --crate-type=proc-macro
//@[rlib_] compile-flags: --crate-type=rlib
//@[staticlib_] compile-flags: --crate-type=staticlib

// The compiler may emit a warning that causes stderr output
// that contains a warning this test does not wish to check.
//@[proc_macro_] needs-unwind

#![crate_name = "NonSnakeCase"]
//[cdylib_,dylib_,lib_,proc_macro_,rlib_,staticlib_]~^ ERROR crate `NonSnakeCase` should have a snake case name
#![deny(non_snake_case)]

fn main() {}
