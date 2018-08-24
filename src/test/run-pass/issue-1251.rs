// pretty-expanded FIXME #23616
// ignore-wasm32-bare no libc to test ffi with

#![feature(libc)]

#![crate_id="rust_get_test_int"]

mod rustrt {
    extern crate libc;

    extern {
        pub fn rust_get_test_int() -> libc::intptr_t;
    }
}

pub fn main() { }
