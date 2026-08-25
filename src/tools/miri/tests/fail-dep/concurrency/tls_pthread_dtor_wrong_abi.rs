//@ignore-target: windows # No pthreads on Windows

use std::{mem, ptr};

pub type Key = libc::pthread_key_t;

pub unsafe fn create(dtor: unsafe fn(*mut u8)) -> Key {
    let mut key = 0;
    assert_eq!(libc::pthread_key_create(&mut key, mem::transmute(dtor)), 0);
    //~^ERROR: calling a function with calling convention "Rust"
    key
}

unsafe fn dtor(_ptr: *mut u8) {}

fn main() {
    unsafe {
        let key = create(dtor);
        libc::pthread_setspecific(key, ptr::without_provenance(1));
    }
}
