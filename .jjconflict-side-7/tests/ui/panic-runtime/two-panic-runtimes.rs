// ignore-tidy-linelength
//@ build-fail
//@ dont-check-compiler-stderr
//@ aux-build:panic-runtime-unwind.rs
//@ aux-build:panic-runtime-unwind2.rs
//@ aux-build:panic-runtime-lang-items.rs

// NOTE: there can be additional errors regarding trying to mix this crate if the precompiled target
// (such as `wasm32-unknown-unknown` currently unconditionally defaulting to panic=abort) panic
// strategy differs to abort, then involving a potentially-unwinding `panic_runtime_unwind` that
// uses a different panic strategy. These errors are important but not to the test intention, which
// is to check that trying to bring two panic runtimes (`panic_runtime_unwind`) and
// (`panic_runtime_unwind2`) is prohibited. As such, the additional errors are not checked in this
// test.
//@ dont-require-annotations: ERROR

#![no_std]
#![no_main]

extern crate panic_runtime_unwind;
extern crate panic_runtime_unwind2;
extern crate panic_runtime_lang_items;

fn main() {}

//~? ERROR cannot link together two panic runtimes: panic_runtime_unwind and panic_runtime_unwind2
