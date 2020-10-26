#![deny(temporary_cstring_as_ptr)]

use std::ffi::CString;

fn some_function(data: *const i8) {}

fn main() {
    some_function(CString::new("").unwrap().as_ptr());
    //~^ ERROR getting the inner pointer of a temporary `CString`
}
