//@ignore-target: windows # No libc

#[path = "../../utils/libc.rs"]
mod libc_utils;

use std::ffi::CStr;
use std::{io, ptr};

use libc_utils::*;

fn main() {
    test_ok();
    test_null_ptr();
}

fn test_ok() {
    // SAFETY: all zeros for `utsname` is valid.
    let mut uname: libc::utsname = unsafe { std::mem::zeroed() };
    errno_check(unsafe { libc::uname(&mut uname) });

    assert_eq!(unsafe { CStr::from_ptr(&uname.sysname as *const _) }, c"Miri");
    assert_eq!(unsafe { CStr::from_ptr(&uname.nodename as *const _) }, c"Miri");
    assert_eq!(
        unsafe { CStr::from_ptr(&uname.release as *const _) }.to_str().unwrap(),
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(unsafe { CStr::from_ptr(&uname.version as *const _) }, c"Miri 0.1.0");
    assert_eq!(
        unsafe { CStr::from_ptr(&uname.machine as *const _) }.to_str().unwrap(),
        std::env::consts::ARCH
    );
    #[cfg(any(target_os = "linux", target_os = "android"))]
    assert_eq!(unsafe { CStr::from_ptr(&uname.domainname as *const _) }, c"(none)");
}

fn test_null_ptr() {
    let err = errno_result(unsafe { libc::uname(ptr::null_mut()) }).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EFAULT));
    assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::EFAULT));
}
