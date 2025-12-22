//@ignore-target: windows # No libc

use std::ffi::CStr;
use std::{io, ptr};

fn main() {
    test_ok();
    test_null_ptr();
}

fn test_ok() {
    // SAFETY: all zeros for `utsname` is valid.
    let mut uname: libc::utsname = unsafe { std::mem::zeroed() };
    let result = unsafe { libc::uname(&mut uname) };
    if result != 0 {
        panic!("failed to call uname");
    }

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
    let result = unsafe { libc::uname(ptr::null_mut()) };
    assert_eq!(result, -1);
    assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::EFAULT));
}
