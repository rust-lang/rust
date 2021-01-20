// run-pass
#![allow(dead_code)]
// ignore-wasm32-bare no libc to test ffi with
// pretty-expanded FIXME #23616

#![feature(rustc_private)]

extern crate libc;

mod bar {
    extern {}
}

mod zed {
    extern {}
}

mod mlibc {
    use libc::{c_int, c_void, size_t, ssize_t};

    extern {
        pub fn write(fd: c_int, buf: *const c_void, count: size_t) -> ssize_t;
    }
}

mod baz {
    extern {}
}

pub fn main() { }
