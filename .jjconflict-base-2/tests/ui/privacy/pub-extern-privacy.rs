//@ run-pass

use std::mem::transmute;

mod a {
    extern "C" {
        pub fn free(x: *mut std::ffi::c_void);
    }
}

pub fn main() {
    unsafe {
        a::free(transmute(0_usize));
    }
}
