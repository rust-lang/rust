// compile-flags:-C panic=unwind
// no-prefer-dynamic

#![feature(panic_runtime)]
#![crate_type = "rlib"]

#![no_std]
#![panic_runtime]

#[no_mangle]
pub extern "C" fn __rust_maybe_catch_panic() {}

#[no_mangle]
pub extern "C" fn __rust_start_panic() {}

#[no_mangle]
pub extern "C" fn rust_eh_personality() {}
