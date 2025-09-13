//@ check-pass
//@ edition: 2021

use std::ffi::c_void;

// From core

#[no_mangle]
pub extern "C" fn memcpy(
    dest: *mut c_void,
    src: *const c_void,
    n: i64,
) -> *mut c_void { std::ptr::null_mut() }
//~^^^^^ WARN `memcpy` clashes

#[no_mangle]
pub fn memmove() {}
//~^ WARN `memmove` clashes

#[no_mangle]
pub fn memset() {}
//~^ WARN `memset` clashes

#[no_mangle]
pub fn memcmp() {}
//~^ WARN `memcmp` clashes

#[export_name = "bcmp"]
pub fn bcmp_() {}
//~^ WARN `bcmp` clashes

#[no_mangle]
pub fn strlen() {}
//~^ WARN `strlen` clashes

// From std

#[no_mangle]
pub fn open() {}
//~^ WARN `open` clashes

#[export_name = "read"]
pub async fn read1() {}
//~^ WARN `read` clashes

#[export_name = "write"]
pub fn write1() {}
//~^ WARN `write` clashes

#[export_name = "close"]
pub fn close_() {}
//~^ WARN `close` clashes

extern "C" {
    // No warning, not a body.
    pub fn close(a: i32) -> i32;
}

fn main() {}
