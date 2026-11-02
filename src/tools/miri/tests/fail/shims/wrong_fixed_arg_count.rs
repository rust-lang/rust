//@ignore-target: windows # File handling is not implemented yet
//@compile-flags: -Zmiri-disable-isolation

#![allow(invalid_runtime_symbol_definitions)]

use std::ffi::{c_char, c_int};

extern "C" {
    fn open(path: *const c_char, ...) -> c_int;
}

fn main() {
    let c_path = c"./text";
    let _fd = unsafe {
        open(c_path.as_ptr(), /* value does not matter */ 0)
        //~^ ERROR: takes 2 fixed (non-variadic) arguments, but 1 argument was given
    };
}
