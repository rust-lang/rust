// ignore-linux: GetProcAddress() is not available on Linux
// ignore-macos: GetProcAddress() is not available on macOS

use std::{ffi::c_void, os::raw::c_char, ptr};

extern "system" {
    fn GetProcAddress(
        hModule: *mut c_void,
        lpProcName: *const c_char,
    ) -> extern "system" fn() -> isize;
}

fn main() {
    unsafe {
        GetProcAddress(ptr::null_mut(), b"foo\0".as_ptr().cast());
        //~^ ERROR unsupported operation: unsupported Windows dlsym: foo
    }
}
