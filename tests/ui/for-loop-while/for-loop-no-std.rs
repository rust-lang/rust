//@ run-pass
//@ ignore-emscripten no no_std executables
//@ ignore-wasm different `main` convention
#![allow(unused_imports)]
#![no_std]
#![no_main]

// Import global allocator and panic handler.
extern crate std as other;

#[macro_use] extern crate alloc;

use alloc::string::ToString;

#[no_mangle]
extern "C" fn main(_argc: core::ffi::c_int, _argv: *const *const u8) -> core::ffi::c_int {
    for _ in [1,2,3].iter() { }
    0
}
