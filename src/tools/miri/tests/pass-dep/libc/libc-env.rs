//@ignore-target: windows # No libc
//@compile-flags: -Zmiri-disable-isolation

use std::ffi::CStr;

fn test_getenv() {
    let s = unsafe { libc::getenv(b"MIRI_ENV_VAR_TEST\0".as_ptr().cast()) };
    assert!(!s.is_null());
    let value = unsafe { CStr::from_ptr(s).to_str().unwrap() };
    assert_eq!(value, "0");

    // Get a non-existing environment variable
    let s = unsafe { libc::getenv(b"MIRI_TEST_NONEXISTENT_VAR\0".as_ptr().cast()) };
    assert!(s.is_null());

    // Empty string should not crash
    let s = unsafe { libc::getenv(b"\0".as_ptr().cast()) };
    assert!(s.is_null());
}

fn test_setenv() {
    // Set a new environment variable
    let result = unsafe {
        libc::setenv(b"MIRI_TEST_VAR\0".as_ptr().cast(), b"test_value\0".as_ptr().cast(), 1)
    };
    assert_eq!(result, 0);

    // Verify it was set
    let s = unsafe { libc::getenv(b"MIRI_TEST_VAR\0".as_ptr().cast()) };
    assert!(!s.is_null());
    let value = unsafe { CStr::from_ptr(s).to_str().unwrap() };
    assert_eq!(value, "test_value");

    // Test overwriting an existing variable
    let result = unsafe {
        libc::setenv(b"MIRI_TEST_VAR\0".as_ptr().cast(), b"new_value\0".as_ptr().cast(), 1)
    };
    assert_eq!(result, 0);

    // Verify it was updated
    let s = unsafe { libc::getenv(b"MIRI_TEST_VAR\0".as_ptr().cast()) };
    assert!(!s.is_null());
    let value = unsafe { CStr::from_ptr(s).to_str().unwrap() };
    assert_eq!(value, "new_value");

    // Test invalid parameters
    let result = unsafe { libc::setenv(std::ptr::null(), b"value\0".as_ptr().cast(), 1) };
    assert_eq!(result, -1);

    let result = unsafe { libc::setenv(b"\0".as_ptr().cast(), b"value\0".as_ptr().cast(), 1) };
    assert_eq!(result, -1);

    let result =
        unsafe { libc::setenv(b"INVALID=NAME\0".as_ptr().cast(), b"value\0".as_ptr().cast(), 1) };
    assert_eq!(result, -1);
}

fn test_unsetenv() {
    // Set a variable
    let result = unsafe {
        libc::setenv(b"MIRI_TEST_UNSET_VAR\0".as_ptr().cast(), b"to_be_unset\0".as_ptr().cast(), 1)
    };
    assert_eq!(result, 0);

    // Verify it exists
    let s = unsafe { libc::getenv(b"MIRI_TEST_UNSET_VAR\0".as_ptr().cast()) };
    assert!(!s.is_null());

    // Unset it
    let result = unsafe { libc::unsetenv(b"MIRI_TEST_UNSET_VAR\0".as_ptr().cast()) };
    assert_eq!(result, 0);

    // Verify it was unset
    let s = unsafe { libc::getenv(b"MIRI_TEST_UNSET_VAR\0".as_ptr().cast()) };
    assert!(s.is_null());

    // Test unsetting a non-existing variable (should succeed)
    let result = unsafe { libc::unsetenv(b"MIRI_TEST_NONEXISTENT_VAR\0".as_ptr().cast()) };
    assert_eq!(result, 0);

    // Test invalid parameters
    let result = unsafe { libc::unsetenv(std::ptr::null()) };
    assert_eq!(result, -1);

    let result = unsafe { libc::unsetenv(b"\0".as_ptr().cast()) };
    assert_eq!(result, -1);

    let result = unsafe { libc::unsetenv(b"INVALID=NAME\0".as_ptr().cast()) };
    assert_eq!(result, -1);
}

fn main() {
    test_getenv();
    test_setenv();
    test_unsetenv();
}
