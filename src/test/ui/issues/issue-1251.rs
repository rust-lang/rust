// build-pass (FIXME(62277): could be check-pass?)
#![allow(unused_attributes)]
#![allow(dead_code)]
// pretty-expanded FIXME #23616
// ignore-wasm32-bare no libc to test ffi with

#![feature(rustc_private)]

#![crate_id="rust_get_test_int"]

mod rustrt {
    extern crate libc;

    extern {
        pub fn rust_get_test_int() -> libc::intptr_t;
    }
}

pub fn main() { }
