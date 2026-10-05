//@ignore-target: windows # File handling is not implemented yet
//@compile-flags: -Zmiri-disable-isolation

#![allow(suspicious_runtime_symbol_definitions)]

use std::ffi::{c_char, c_int};

extern "C" {
    fn open(path: *const c_char, oflag: c_int, ...) -> c_int;
    // correct fd type is i32
    fn close(fd: u32) -> c_int;
}

fn main() {
    let c_path = c"./text";
    let fd = unsafe {
        open(c_path.as_ptr(), /* value does not matter */ 0)
    } as u32;
    let _ = unsafe {
        close(fd);
        //~^ ERROR: type i32 passing argument of type u32
    };
}
