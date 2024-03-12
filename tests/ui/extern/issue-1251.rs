//@ build-pass
#![allow(unused_attributes)]
#![allow(dead_code)]
//@ pretty-expanded FIXME #23616
#![feature(rustc_private)]

mod rustrt {
    extern crate libc;

    extern "C" {
        pub fn rust_get_test_int() -> libc::intptr_t;
    }
}

pub fn main() {}
