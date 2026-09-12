// This test makes sure that the following symbols are not linted against
// in a #[no_std] context.

//@ check-pass

#![no_std]
#![crate_type = "lib"]

use core::ffi::{c_char, c_int, c_void};

#[no_mangle]
pub fn open() {}

extern "C" {
    pub fn read();
    pub fn write();
}

#[no_mangle]
pub static close: () = ();

extern "C" {
    pub fn malloc();
    pub fn realloc();
    pub fn free();
    pub fn exit();
}
