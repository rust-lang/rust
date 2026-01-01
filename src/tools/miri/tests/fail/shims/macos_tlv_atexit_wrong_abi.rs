//@only-target: darwin

use std::{mem, ptr};

extern "C" {
    fn _tlv_atexit(dtor: unsafe extern "C" fn(*mut u8), arg: *mut u8);
}

fn register(dtor: unsafe fn(*mut u8)) {
    unsafe {
        _tlv_atexit(mem::transmute(dtor), ptr::null_mut());
        //~^ERROR: calling a function with calling convention "Rust"
    }
}

fn main() {
    register(|_| ());
}
