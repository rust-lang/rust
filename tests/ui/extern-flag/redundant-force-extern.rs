// check-pass
// ignore-cross-compile (needs dylibs and compiletest doesn't have a more specific header)
// aux-crate:force:panic_handler=panic_handler.rs
// compile-flags: -Zunstable-options --crate-type dylib
// edition:2018

#![no_std]

extern crate panic_handler;

fn foo() {}
