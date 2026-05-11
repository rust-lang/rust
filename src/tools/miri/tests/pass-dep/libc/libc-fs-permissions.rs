//@ignore-target: windows # no libc
//@ignore-host: windows # needs unix PermissionExt
//@compile-flags: -Zmiri-disable-isolation

#![feature(io_error_more)]
#![feature(io_error_uncategorized)]

use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::os::unix::ffi::OsStrExt;

#[path = "../../utils/mod.rs"]
mod utils;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::{errno_check, errno_result};

fn main() {
    test_chmod();
    test_fchmod();
}

#[track_caller]
fn getmod(path: &CStr) -> u32 {
    let mut stat = MaybeUninit::<libc::stat>::uninit();
    unsafe { errno_check(libc::stat(path.as_ptr(), stat.as_mut_ptr())) };
    u32::from(unsafe { stat.assume_init_ref().st_mode & !libc::S_IFMT })
}

fn test_chmod() {
    let path = utils::prepare_with_content("miri_test_libc_chmod.txt", b"abcdef");
    let c_path = CString::new(path.as_os_str().as_bytes()).expect("CString::new failed");

    unsafe { errno_check(libc::chmod(c_path.as_ptr(), 0o777)) };
    assert_eq!(getmod(&c_path), 0o777);
    unsafe { errno_check(libc::chmod(c_path.as_ptr(), 0o610)) };
    assert_eq!(getmod(&c_path), 0o610);
}

fn test_fchmod() {
    let path = utils::prepare_with_content("miri_test_libc_chmod.txt", b"abcdef");
    let c_path = CString::new(path.as_os_str().as_bytes()).expect("CString::new failed");

    let fd = unsafe { errno_result(libc::open(c_path.as_ptr(), libc::O_RDONLY)).unwrap() };
    unsafe { errno_check(libc::fchmod(fd, 0o777)) };
    assert_eq!(getmod(&c_path), 0o777);
    unsafe { errno_check(libc::fchmod(fd, 0o610)) };
    assert_eq!(getmod(&c_path), 0o610);
}
