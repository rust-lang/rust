//ignore-windows

#![feature(libc)]
extern crate libc;

use std::mem;

pub type Key = libc::pthread_key_t;

pub unsafe fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
    let mut key = 0;
    assert_eq!(libc::pthread_key_create(&mut key, mem::transmute(dtor)), 0);
    key
}

fn main() {
    let _ = unsafe { create(None) };
}
