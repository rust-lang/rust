//@ignore-target: windows # File handling is not implemented yet
//@compile-flags: -Zmiri-disable-isolation

#![allow(invalid_runtime_symbol_definitions)]

use std::ffi::{c_char, c_int};

// Declare a variadic function as non-variadic.
extern "C" {
    fn open(path: *const c_char, oflag: c_int) -> c_int;
}

fn main() {
    let c_path = c"./text";
    let _fd = unsafe {
        open(c_path.as_ptr(), /* value does not matter */ 0)
        //~^ ERROR: is a c-variadic function, but the caller is using a non-variadic signature
    };
}
