// this program is not technically incorrect, but is an obscure enough style to be worth linting
#![deny(temporary_cstring_as_ptr)]
//~^ WARNING lint `temporary_cstring_as_ptr` has been renamed to `dangling_pointers_from_temporaries`

use std::ffi::CString;

macro_rules! mymacro {
    () => {
        let s = CString::new("some text").unwrap().as_ptr();
        //~^ ERROR getting a pointer from a temporary `CString` will result in a dangling pointer
    }
}

fn main() {
    let s = CString::new("some text").unwrap().as_ptr();
    //~^ ERROR getting a pointer from a temporary `CString` will result in a dangling pointer
    mymacro!();
}
