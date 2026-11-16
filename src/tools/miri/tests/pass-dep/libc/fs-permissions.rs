//@ignore-target: windows # libc bits exist, but we don't support them
//@ignore-host: windows # on Windows hosts we only supports 0o444 and 0o666
//@compile-flags: -Zmiri-disable-isolation
//@run-native

use std::ffi::CStr;
use std::mem::MaybeUninit;

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
fn getmod(path: &CStr) -> libc::mode_t {
    let mut stat = MaybeUninit::<libc::stat>::uninit();
    unsafe { errno_check(libc::stat(path.as_ptr(), stat.as_mut_ptr())) };
    unsafe { stat.assume_init_ref().st_mode }
}

fn test_chmod() {
    let path = utils::prepare_with_content("miri_test_libc_chmod.txt", b"abcdef");
    let c_path = utils::into_c_string(path);

    unsafe { errno_check(libc::chmod(c_path.as_ptr(), 0o777)) };
    assert_eq!(getmod(&c_path) & !libc::S_IFMT, 0o777);
    unsafe { errno_check(libc::chmod(c_path.as_ptr(), 0o610)) };
    assert_eq!(getmod(&c_path) & !libc::S_IFMT, 0o610);
    // We can leave the file type in the mode that we pass to `chmod`.
    unsafe { errno_check(libc::chmod(c_path.as_ptr(), getmod(&c_path) | 0o111)) };
    assert_eq!(getmod(&c_path) & !libc::S_IFMT, 0o711);
}

fn test_fchmod() {
    let path = utils::prepare_with_content("miri_test_libc_chmod.txt", b"abcdef");
    let c_path = utils::into_c_string(path);

    let fd = unsafe { errno_result(libc::open(c_path.as_ptr(), libc::O_RDONLY)).unwrap() };
    unsafe { errno_check(libc::fchmod(fd, 0o777)) };
    assert_eq!(getmod(&c_path) & !libc::S_IFMT, 0o777);
    unsafe { errno_check(libc::fchmod(fd, 0o610)) };
    assert_eq!(getmod(&c_path) & !libc::S_IFMT, 0o610);
    // We can leave the file type in the mode that we pass to `fchmod`.
    unsafe { errno_check(libc::fchmod(fd, getmod(&c_path) | 0o111)) };
    assert_eq!(getmod(&c_path) & !libc::S_IFMT, 0o711);
}
