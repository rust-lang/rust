//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-isolation-error=warn-nobacktrace
//@revisions: tcp udp

use std::io::ErrorKind;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

const SOCKET_TYPE: libc::c_int = cfg_select! {
    tcp => libc::SOCK_STREAM,
    udp => libc::SOCK_DGRAM,
};

fn main() {
    unsafe {
        let err = errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::PermissionDenied);
        // check that it is the right kind of `PermissionDenied`
        assert_eq!(err.raw_os_error(), Some(libc::EACCES));
    }
}
