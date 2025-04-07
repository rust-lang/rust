//@ignore-target: windows # File handling is not implemented yet
//@compile-flags: -Zmiri-disable-isolation
use std::ffi::{CString, OsStr, c_char, c_int};
use std::os::unix::ffi::OsStrExt;

extern "C" {
    fn open(path: *const c_char, oflag: c_int, ...) -> c_int;
    // correct fd type is i32
    fn close(fd: u32) -> c_int;
}

fn main() {
    let c_path = CString::new(OsStr::new("./text").as_bytes()).expect("CString::new failed");
    let fd = unsafe {
        open(c_path.as_ptr(), /* value does not matter */ 0)
    } as u32;
    let _ = unsafe {
        close(fd);
        //~^ ERROR: calling a function with argument of type i32 passing data of type u32
    };
}
