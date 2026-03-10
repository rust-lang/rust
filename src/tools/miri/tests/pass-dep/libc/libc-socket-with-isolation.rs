//@ignore-target: windows # No libc socket on Windows
//@ignore-target: solaris # Socket is a macro for __xnet7_socket which has no shim
//@ignore-target: illumos # Socket is a macro for __xnet7_socket which has no shim
//@compile-flags: -Zmiri-isolation-error=warn-nobacktrace

use std::io::ErrorKind;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

fn main() {
    unsafe {
        let err = errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::PermissionDenied);
        // check that it is the right kind of `PermissionDenied`
        assert_eq!(err.raw_os_error(), Some(libc::EACCES));
    }
}
