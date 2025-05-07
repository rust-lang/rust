//@ run-pass
//@ ignore-emscripten no no_std executables
//@ ignore-wasm different `main` convention

#![no_std]
#![no_main]

// Import global allocator and panic handler.
extern crate std as other;

#[macro_use] extern crate alloc;

use alloc::vec::Vec;

// Issue #16806

#[no_mangle]
extern "C" fn main(_argc: core::ffi::c_int, _argv: *const *const u8) -> core::ffi::c_int {
    let x: Vec<u8> = vec![0, 1, 2];
    match x.last() {
        Some(&2) => (),
        _ => panic!(),
    }
    0
}
