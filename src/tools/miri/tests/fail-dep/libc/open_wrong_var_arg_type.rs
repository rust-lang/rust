//@ignore-target: windows # File handling is not implemented yet
//@compile-flags: -Zmiri-disable-isolation
//@normalize-stderr-test: "to have type `(u32|i32)`" -> "to have type `$$TYPE`"

#![allow(invalid_runtime_symbol_definitions)]

use std::ffi::{CString, OsStr};
use std::os::unix::ffi::OsStrExt;

use libc::open;

fn main() {
    let c_path = CString::new(OsStr::new("./text").as_bytes()).expect("CString::new failed");
    let _fd = unsafe {
        open(c_path.as_ptr(), libc::O_CREAT, /* should be mode_t */ 0u64)
        //~^ ERROR: /expected argument #3 to have type `(u32|i32)` but got incompatible type `u64`/
    };
}
