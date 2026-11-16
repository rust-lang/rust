//@ignore-target: windows # No libc
//@compile-flags: -Zmiri-disable-isolation
//@run-native

#[path = "../../utils/libc.rs"]
mod libc_utils;

use std::ffi::CStr;
use std::os::unix::ffi::OsStrExt;

use libc_utils::{errno, errno_check, errno_result};

fn test_getenv() {
    let s = unsafe { libc::getenv(c"MIRI_ENV_VAR_TEST".as_ptr()) };
    assert!(!s.is_null());
    let value = unsafe { CStr::from_ptr(s).to_str().unwrap() };
    assert_eq!(value, "0");

    // Get a non-existing environment variable
    let s = unsafe { libc::getenv(c"MIRI_TEST_NONEXISTENT_VAR".as_ptr()) };
    assert!(s.is_null());

    // Empty string should not crash
    let s = unsafe { libc::getenv(c"".as_ptr()) };
    assert!(s.is_null());
}

fn test_setenv() {
    // Set a new environment variable
    let result = unsafe { libc::setenv(c"MIRI_TEST_VAR".as_ptr(), c"test_value".as_ptr(), 1) };
    assert_eq!(result, 0);

    // Verify it was set
    let s = unsafe { libc::getenv(c"MIRI_TEST_VAR".as_ptr()) };
    assert!(!s.is_null());
    let value = unsafe { CStr::from_ptr(s).to_str().unwrap() };
    assert_eq!(value, "test_value");

    // Test overwriting an existing variable
    let result = unsafe { libc::setenv(c"MIRI_TEST_VAR".as_ptr(), c"new_value".as_ptr(), 1) };
    assert_eq!(result, 0);

    // Verify it was updated
    let s = unsafe { libc::getenv(c"MIRI_TEST_VAR".as_ptr()) };
    assert!(!s.is_null());
    let value = unsafe { CStr::from_ptr(s).to_str().unwrap() };
    assert_eq!(value, "new_value");

    // Test invalid parameters
    let result = unsafe { libc::setenv(std::ptr::null(), c"value".as_ptr(), 1) };
    assert_eq!(result, -1);

    let result = unsafe { libc::setenv(c"".as_ptr(), c"value".as_ptr(), 1) };
    assert_eq!(result, -1);

    let result = unsafe { libc::setenv(c"INVALID=NAME".as_ptr(), c"value".as_ptr(), 1) };
    assert_eq!(result, -1);
}

fn test_unsetenv() {
    // Set a variable
    let result =
        unsafe { libc::setenv(c"MIRI_TEST_UNSET_VAR".as_ptr(), c"to_be_unset".as_ptr(), 1) };
    assert_eq!(result, 0);

    // Verify it exists
    let s = unsafe { libc::getenv(c"MIRI_TEST_UNSET_VAR".as_ptr()) };
    assert!(!s.is_null());

    // Unset it
    let result = unsafe { libc::unsetenv(c"MIRI_TEST_UNSET_VAR".as_ptr()) };
    assert_eq!(result, 0);

    // Verify it was unset
    let s = unsafe { libc::getenv(c"MIRI_TEST_UNSET_VAR".as_ptr()) };
    assert!(s.is_null());

    // Test unsetting a non-existing variable (should succeed)
    let result = unsafe { libc::unsetenv(c"MIRI_TEST_NONEXISTENT_VAR".as_ptr()) };
    assert_eq!(result, 0);

    // Test invalid parameters
    let result = unsafe { libc::unsetenv(std::ptr::null()) };
    assert_eq!(result, -1);

    let result = unsafe { libc::unsetenv(c"".as_ptr()) };
    assert_eq!(result, -1);

    let result = unsafe { libc::unsetenv(c"INVALID=NAME".as_ptr()) };
    assert_eq!(result, -1);
}

fn test_getcwd() {
    // Filling the buffer with `1` prevents the null terminator check from passing by mistake
    let mut buf = [1u8; libc::PATH_MAX as usize];

    let result = unsafe { libc::getcwd(buf.as_mut_ptr().cast(), buf.len()) };
    assert_eq!(result, buf.as_mut_ptr().cast());

    let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
    assert_eq!(cstr.to_bytes(), std::env::current_dir().unwrap().as_os_str().as_bytes());

    let len = cstr.count_bytes();

    // ERANGE: The pathname fits into the buffer,
    // but there isn't enough space for the `\0` terminator
    let result = unsafe { libc::getcwd(buf.as_mut_ptr().cast(), len) };
    assert!(result.is_null());
    assert_eq!(errno(), libc::ERANGE);

    // EINVAL: The `size` is zero and `buf` is not NULL
    let result = unsafe { libc::getcwd(buf.as_mut_ptr().cast(), 0) };
    assert!(result.is_null());
    assert_eq!(errno(), libc::EINVAL);
}

fn test_chdir() {
    let old_cwd = std::env::current_dir().unwrap();

    errno_check(unsafe { libc::chdir(c"..".as_ptr()) });

    let parent = old_cwd.parent().unwrap_or(&old_cwd);
    assert_eq!(std::env::current_dir().unwrap(), parent);

    // If the cwd isn't restored, subsequent tests would run in the wrong directory
    std::env::set_current_dir(old_cwd).unwrap();

    let err = errno_result(unsafe { libc::chdir(c"thisdoesnotexist".as_ptr()) }).unwrap_err();
    assert_eq!(err.raw_os_error().unwrap(), libc::ENOENT);
}

fn main() {
    test_getenv();
    test_setenv();
    test_unsetenv();
    test_getcwd();
    test_chdir();
}
