// ignore-tidy-linelength

use std::ffi::CString;

fn main() {
    let s = CString::new("some text").unwrap().as_ptr(); //~ ERROR getting the inner pointer of a temporary `CString`
}
