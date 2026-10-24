//@ aux-crate:panic_handler=panic_handler.rs
//@ ignore-cross-compile (needs dylibs and compiletest doesn't have a more specific header)
// compile_flags: -Zunstable-options --crate-type dylib
//@ dont-check-compiler-stderr
//@ edition: 2018

#![no_std]

fn foo() {} //~ ERROR `main` function not found in crate `no_force_extern`

//~? ERROR `#[panic_handler]` function required, but not found
//~? ERROR unwinding panics are not supported without std
