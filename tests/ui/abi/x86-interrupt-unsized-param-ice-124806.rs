//! Regression test for <https://github.com/rust-lang/rust/issues/124806>
//!
//! `extern "x86-interrupt"` passes its parameters `byval`, so they must be
//!   `Sized` even with `unsized_fn_params` enabled. An unsized parameter once
//!   reached code generation and caused an ICE.
//@ add-minicore
//@ revisions: x64 i686
//
//@ [x64] needs-llvm-components: x86
//@ [x64] compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib
//@ [i686] needs-llvm-components: x86
//@ [i686] compile-flags: --target=i686-unknown-linux-gnu --crate-type=rlib
//@ ignore-backends: gcc
#![no_core]
#![feature(no_core, abi_x86_interrupt, unsized_fn_params)]

extern crate minicore;
use minicore::*;

extern "x86-interrupt" fn interrupt_unsized(_a: str) {}
//~^ ERROR the size for values of type `str` cannot be known at compilation time
//~| NOTE argument required to be sized due to `extern "x86-interrupt"` ABI
//~| NOTE doesn't have a size known at compile-time
