//@ build-pass
#![allow(unused_attributes)]
#![allow(dead_code)]

mod rustrt {
    extern "C" {
        pub fn rust_get_test_int() -> isize;
    }
}

pub fn main() {}
