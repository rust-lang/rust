// this program is not technically incorrect, but is an obscure enough style to be worth linting
#![deny(temporary_cstring_as_ptr)]

use std::ffi::CString;

macro_rules! mymacro {
    () => {
        let s = CString::new("some text").unwrap().as_ptr();
        //~^ ERROR this pointer will immediately dangle
    }
}

fn main() {
    let s = CString::new("some text").unwrap().as_ptr();
    //~^ ERROR this pointer will immediately dangle
    mymacro!();
}
