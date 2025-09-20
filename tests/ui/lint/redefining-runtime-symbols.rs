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
//~^^^^^ WARN redefinition of the runtime `memcpy` symbol

#[no_mangle]
pub fn memmove() {}
//~^ WARN redefinition of the runtime `memmove` symbol

#[no_mangle]
pub fn memset() {}
//~^ WARN redefinition of the runtime `memset` symbol

#[no_mangle]
pub fn memcmp() {}
//~^ WARN redefinition of the runtime `memcmp` symbol

#[export_name = "bcmp"]
pub fn bcmp_() {}
//~^ WARN redefinition of the runtime `bcmp` symbol

#[no_mangle]
pub static strlen: () = ();
//~^ WARN redefinition of the runtime `strlen` symbol

// From std

#[no_mangle]
pub fn open() {}
//~^ WARN redefinition of the runtime `open` symbol

#[no_mangle]
pub fn open64() {}
//~^ WARN redefinition of the runtime `open64` symbol

#[export_name = "read"]
pub async fn read1() {}
//~^ WARN redefinition of the runtime `read` symbol

#[export_name = "write"]
pub fn write1() {}
//~^ WARN redefinition of the runtime `write` symbol

#[export_name = "close"]
pub fn close_() {}
//~^ WARN redefinition of the runtime `close` symbol

extern "C" {
    // No warning, not a body.
    pub fn close(a: i32) -> i32;
}

fn main() {}
