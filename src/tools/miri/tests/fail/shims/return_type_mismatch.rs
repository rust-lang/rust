//@ignore-target: windows # File handling is not implemented yet
//@compile-flags: -Zmiri-disable-isolation
use std::ffi::{CString, OsStr, c_char, c_int, c_short};
use std::os::unix::ffi::OsStrExt;

extern "C" {
    fn open(path: *const c_char, oflag: c_int, ...) -> c_int;
    // correct return type is i32
    fn close(fd: c_int) -> c_short;
}

fn main() {
    let c_path = CString::new(OsStr::new("./text").as_bytes()).expect("CString::new failed");
    let fd = unsafe {
        open(c_path.as_ptr(), /* value does not matter */ 0)
    };
    let _ = unsafe {
        close(fd);
        //~^ ERROR: calling a function with return type i32 passing return place of type i16
    };
}
