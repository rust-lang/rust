//@ignore-target: windows # File handling is not implemented yet
//@compile-flags: -Zmiri-disable-isolation
use std::ffi::{CString, OsStr, c_char, c_int};
use std::os::unix::ffi::OsStrExt;

extern "C" {
    fn open(path: *const c_char, ...) -> c_int;
}

fn main() {
    let c_path = CString::new(OsStr::new("./text").as_bytes()).expect("CString::new failed");
    let _fd = unsafe {
        open(c_path.as_ptr(), /* value does not matter */ 0)
        //~^ ERROR: incorrect number of fixed arguments for variadic function
    };
}
