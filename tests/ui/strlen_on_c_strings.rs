#![warn(clippy::strlen_on_c_strings)]
#![allow(dead_code)]
#![feature(rustc_private)]
extern crate libc;

use std::ffi::{CStr, CString};

fn main() {
    // CString
    let cstring = CString::new("foo").expect("CString::new failed");
    let len = unsafe { libc::strlen(cstring.as_ptr()) };

    // CStr
    let cstr = CStr::from_bytes_with_nul(b"foo\0").expect("CStr::from_bytes_with_nul failed");
    let len = unsafe { libc::strlen(cstr.as_ptr()) };
}
