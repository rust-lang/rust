//@ check-pass

#![deny(temporary_cstring_as_ptr)]
//~^ WARNING lint `temporary_cstring_as_ptr` has been renamed to `dangling_pointers_from_temporaries`

use std::ffi::CString;
use std::os::raw::c_char;

fn some_function(data: *const c_char) {}

fn main() {
    some_function(CString::new("").unwrap().as_ptr());
}
