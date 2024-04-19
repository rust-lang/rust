//@ build-pass
#![allow(unused_attributes)]
#![allow(dead_code)]
//@ pretty-expanded FIXME #23616

mod rustrt {
    extern "C" {
        pub fn rust_get_test_int() -> isize;
    }
}

pub fn main() {}
