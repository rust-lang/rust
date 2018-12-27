// run-pass
// ignore-wasm32-bare no libc to test ffi with

// pretty-expanded FIXME #23616

use std::mem::transmute;

mod a {
    extern {
        pub fn free(x: *const u8);
    }
}

pub fn main() {
    unsafe {
        a::free(transmute(0_usize));
    }
}
