// this program is not technically incorrect, but is an obscure enough style to be worth linting
#![deny(temporary_cstring_as_ptr)]
//~^ WARNING lint `temporary_cstring_as_ptr` has been renamed to `dangling_pointers_from_temporaries`

use std::ffi::CString;

macro_rules! mymacro {
    () => {
        let s = CString::new("some text").unwrap().as_ptr();
        //~^ ERROR a dangling pointer will be produced because the temporary `CString` will be dropped
    }
}

fn main() {
    let s = CString::new("some text").unwrap().as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `CString` will be dropped
    mymacro!();
}
