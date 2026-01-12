#![warn(clippy::strlen_on_c_strings)]
#![allow(clippy::manual_c_str_literals)]

use libc::strlen;
use std::ffi::{CStr, CString};

fn main() {
    // CString
    let cstring = CString::new("foo").expect("CString::new failed");
    let _ = unsafe { libc::strlen(cstring.as_ptr()) };
    //~^ ERROR: using `libc::strlen` on a `CString` value

    // CStr
    let cstr = CStr::from_bytes_with_nul(b"foo\0").expect("CStr::from_bytes_with_nul failed");
    let _ = unsafe { libc::strlen(cstr.as_ptr()) };
    //~^ ERROR: using `libc::strlen` on a `CStr` value

    let _ = unsafe { strlen(cstr.as_ptr()) };
    //~^ ERROR: using `libc::strlen` on a `CStr` value

    let pcstr: *const &CStr = &cstr;
    let _ = unsafe { strlen((*pcstr).as_ptr()) };
    //~^ ERROR: using `libc::strlen` on a `CStr` value

    unsafe fn unsafe_identity<T>(x: T) -> T {
        x
    }
    let _ = unsafe { strlen(unsafe_identity(cstr).as_ptr()) };
    //~^ ERROR: using `libc::strlen` on a `CStr` value
    let _ = unsafe { strlen(unsafe { unsafe_identity(cstr) }.as_ptr()) };
    //~^ ERROR: using `libc::strlen` on a `CStr` value

    let f: unsafe fn(_) -> _ = unsafe_identity;
    let _ = unsafe { strlen(f(cstr).as_ptr()) };
    //~^ ERROR: using `libc::strlen` on a `CStr` value
}
