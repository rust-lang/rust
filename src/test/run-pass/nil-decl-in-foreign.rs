#![allow(improper_ctypes)]
#![allow(dead_code)]
// Issue #901
// pretty-expanded FIXME #23616

mod libc {
    extern {
        pub fn printf(x: ());
    }
}

pub fn main() { }
