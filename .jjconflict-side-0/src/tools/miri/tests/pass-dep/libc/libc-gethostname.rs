//@ignore-target: windows # No libc
//@run-native

#[path = "../../utils/libc.rs"]
mod libc_utils;

use std::ffi::CStr;

use libc_utils::*;

fn main() {
    test_ok();

    test_too_small();
}

fn get_hostname() -> Vec<u8> {
    let mut name = [0u8; 1024]; // We assume this is enough space.
    errno_check(unsafe { libc::gethostname(name.as_mut_ptr().cast(), name.len()) });

    let hostname = unsafe { CStr::from_ptr(name.as_ptr().cast()) }.to_bytes().to_owned();
    assert!(!hostname.is_empty());
    hostname
}

fn test_ok() {
    let hostname = get_hostname();

    if cfg!(miri) {
        assert_eq!(hostname, b"Miri");
    }
}

fn test_too_small() {
    let hostname = get_hostname();

    // Leave no room for the trailing null byte.
    let mut name = vec![0u8; hostname.len()];
    let result = unsafe { libc::gethostname(name.as_mut_ptr().cast(), name.len()) };
    cfg_select! {
        target_os = "android" => {
            let err = errno_result(result).unwrap_err();
            assert!(name.iter().all(|byte| *byte == 0));
            assert_eq!(err.raw_os_error(), Some(libc::ENAMETOOLONG));
        }
        any(all(target_os = "linux", target_env = "gnu"), target_os = "freebsd") => {
            let err = errno_result(result).unwrap_err();
            assert_eq!(name, hostname);
            assert_eq!(err.raw_os_error(), Some(libc::ENAMETOOLONG));
        }
        any(
            all(target_os = "linux", not(target_env = "gnu")),
            target_os = "macos",
            target_os = "illumos",
            target_os = "solaris",
        ) => {
            errno_check(result);
            let mut expected = hostname.to_owned();
            *expected.last_mut().unwrap() = 0;
            assert_eq!(name, expected);
        }
        _ => {
            compile_error!("unsupported target");
        }
    }
}
