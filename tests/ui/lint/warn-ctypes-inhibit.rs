//@ run-pass
//@ compile-flags:-D improper-ctypes

#![allow(dead_code)]
#![allow(improper_ctypes)]
#![allow(suspicious_runtime_symbol_definitions)]

mod libc {
    extern "C" {
        pub fn malloc(size: isize) -> *const u8;
    }
}

pub fn main() {}
