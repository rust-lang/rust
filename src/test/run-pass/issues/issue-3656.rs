// run-pass
#![allow(dead_code)]
#![allow(improper_ctypes)]

// Issue #3656
// Incorrect struct size computation in the FFI, because of not taking
// the alignment of elements into account.

// pretty-expanded FIXME #23616
// ignore-wasm32-bare no libc to test with

#![feature(rustc_private)]

extern crate libc;
use libc::{c_uint, uint32_t, c_void};

pub struct KEYGEN {
    hash_algorithm: [c_uint; 2],
    count: uint32_t,
    salt: *const c_void,
    salt_size: uint32_t,
}

extern {
    // Bogus signature, just need to test if it compiles.
    pub fn malloc(data: KEYGEN);
}

pub fn main() {
}
