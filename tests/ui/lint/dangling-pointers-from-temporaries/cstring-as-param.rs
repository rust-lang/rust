#![deny(temporary_cstring_as_ptr)]
//~^ [renamed_and_removed_lints]

use std::ffi::CString;
use std::os::raw::c_char;

fn some_function(data: *const c_char) {}

fn main() {
    some_function(CString::new("").unwrap().as_ptr());
    //~^ ERROR getting a pointer from a temporary `CString` will result in a dangling pointer
}
