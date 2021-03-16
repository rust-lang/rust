// ignore-windows: No dlsym() on Windows

#![feature(rustc_private)]

extern crate libc;

use std::ptr;

fn main() {
    unsafe {
        libc::dlsym(ptr::null_mut(), b"foo\0".as_ptr().cast());
        //~^ ERROR unsupported operation: unsupported
    }
}
