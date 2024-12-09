// Symlink tests are separate since they don't in general work on a Windows host.
//@ignore-host: windows # creating symlinks requires admin permissions on Windows
//@ignore-target: windows # File handling is not implemented yet
//@compile-flags: -Zmiri-disable-isolation

use std::ffi::CString;
use std::io::{Error, ErrorKind};
use std::os::unix::ffi::OsStrExt;

#[path = "../../utils/mod.rs"]
mod utils;

fn main() {
    test_readlink();
    test_nofollow_symlink();
}

fn test_readlink() {
    let bytes = b"Hello, World!\n";
    let path = utils::prepare_with_content("miri_test_fs_link_target.txt", bytes);
    let expected_path = path.as_os_str().as_bytes();

    let symlink_path = utils::prepare("miri_test_fs_symlink.txt");
    std::os::unix::fs::symlink(&path, &symlink_path).unwrap();

    // Test that the expected string gets written to a buffer of proper
    // length, and that a trailing null byte is not written.
    let symlink_c_str = CString::new(symlink_path.as_os_str().as_bytes()).unwrap();
    let symlink_c_ptr = symlink_c_str.as_ptr();

    // Make the buf one byte larger than it needs to be,
    // and check that the last byte is not overwritten.
    let mut large_buf = vec![0xFF; expected_path.len() + 1];
    let res =
        unsafe { libc::readlink(symlink_c_ptr, large_buf.as_mut_ptr().cast(), large_buf.len()) };
    // Check that the resolved path was properly written into the buf.
    assert_eq!(&large_buf[..(large_buf.len() - 1)], expected_path);
    assert_eq!(large_buf.last(), Some(&0xFF));
    assert_eq!(res, large_buf.len() as isize - 1);

    // Test that the resolved path is truncated if the provided buffer
    // is too small.
    let mut small_buf = [0u8; 2];
    let res =
        unsafe { libc::readlink(symlink_c_ptr, small_buf.as_mut_ptr().cast(), small_buf.len()) };
    assert_eq!(small_buf, &expected_path[..small_buf.len()]);
    assert_eq!(res, small_buf.len() as isize);

    // Test that we report a proper error for a missing path.
    let res = unsafe {
        libc::readlink(
            c"MIRI_MISSING_FILE_NAME".as_ptr(),
            small_buf.as_mut_ptr().cast(),
            small_buf.len(),
        )
    };
    assert_eq!(res, -1);
    assert_eq!(Error::last_os_error().kind(), ErrorKind::NotFound);
}

fn test_nofollow_symlink() {
    let bytes = b"Hello, World!\n";
    let path = utils::prepare_with_content("test_nofollow_symlink_target.txt", bytes);

    let symlink_path = utils::prepare("test_nofollow_symlink.txt");
    std::os::unix::fs::symlink(&path, &symlink_path).unwrap();

    let symlink_cpath = CString::new(symlink_path.as_os_str().as_bytes()).unwrap();

    let ret = unsafe { libc::open(symlink_cpath.as_ptr(), libc::O_NOFOLLOW | libc::O_CLOEXEC) };
    assert_eq!(ret, -1);
    let err = Error::last_os_error().raw_os_error().unwrap();
    assert_eq!(err, libc::ELOOP);
}
