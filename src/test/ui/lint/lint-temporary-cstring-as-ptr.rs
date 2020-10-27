// this program is not technically incorrect, but is an obscure enough style to be worth linting
#![deny(temporary_cstring_as_ptr)]

use std::ffi::CString;

fn main() {
    let s = CString::new("some text").unwrap().as_ptr();
    //~^ ERROR getting the inner pointer of a temporary `CString`
}
