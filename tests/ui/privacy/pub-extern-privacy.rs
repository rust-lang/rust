//@ run-pass

//@ pretty-expanded FIXME #23616

use std::mem::transmute;

mod a {
    extern "C" {
        pub fn free(x: *const u8);
    }
}

pub fn main() {
    unsafe {
        a::free(transmute(0_usize));
    }
}
