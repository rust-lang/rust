#![warn(clippy::strlen_on_c_strings)]
#![allow(dead_code)]
#![feature(rustc_private)]
extern crate libc;

use libc::strlen;
use std::ffi::{CStr, CString};

fn main() {
    // CString
    let cstring = CString::new("foo").expect("CString::new failed");
    let len = unsafe { libc::strlen(cstring.as_ptr()) };

    // CStr
    let cstr = CStr::from_bytes_with_nul(b"foo\0").expect("CStr::from_bytes_with_nul failed");
    let len = unsafe { libc::strlen(cstr.as_ptr()) };

    let len = unsafe { strlen(cstr.as_ptr()) };

    let pcstr: *const &CStr = &cstr;
    let len = unsafe { strlen((*pcstr).as_ptr()) };

    unsafe fn unsafe_identity<T>(x: T) -> T {
        x
    }
    let len = unsafe { strlen(unsafe_identity(cstr).as_ptr()) };
    let len = unsafe { strlen(unsafe { unsafe_identity(cstr) }.as_ptr()) };

    let f: unsafe fn(_) -> _ = unsafe_identity;
    let len = unsafe { strlen(f(cstr).as_ptr()) };
}
