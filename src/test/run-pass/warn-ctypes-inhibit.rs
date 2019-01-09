#![allow(dead_code)]
// compile-flags:-D improper-ctypes

// pretty-expanded FIXME #23616

#![allow(improper_ctypes)]

mod libc {
    extern {
        pub fn malloc(size: isize) -> *const u8;
    }
}

pub fn main() {
}
