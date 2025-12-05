//! Test format! macro functionality in no_std environment

//@ run-pass
//@ ignore-emscripten no no_std executables
//@ ignore-wasm different `main` convention

#![feature(lang_items)]
#![no_std]
#![no_main]

// Import global allocator and panic handler.
extern crate std as other;

#[macro_use]
extern crate alloc;

use alloc::string::ToString;

#[no_mangle]
extern "C" fn main(_argc: core::ffi::c_int, _argv: *const *const u8) -> core::ffi::c_int {
    let s = format!("{}", 1_isize);
    assert_eq!(s, "1".to_string());

    let s = format!("test");
    assert_eq!(s, "test".to_string());

    let s = format!("{test}", test = 3_isize);
    assert_eq!(s, "3".to_string());

    let s = format!("hello {}", "world");
    assert_eq!(s, "hello world".to_string());

    0
}
