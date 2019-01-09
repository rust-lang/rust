// compile-flags:-C panic=abort
// no-prefer-dynamic

#![feature(panic_runtime)]
#![crate_type = "rlib"]

#![no_std]
#![panic_runtime]

#[no_mangle]
pub extern fn __rust_maybe_catch_panic() {}

#[no_mangle]
pub extern fn __rust_start_panic() {}

#[no_mangle]
pub extern fn rust_eh_personality() {}
