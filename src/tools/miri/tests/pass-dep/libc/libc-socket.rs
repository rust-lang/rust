//@ignore-target: windows # No libc socket on Windows
//@ignore-target: solaris # Does socket is a macro for __xnet7_socket which has no shim
//@ignore-target: illumos # Does socket is a macro for __xnet7_socket which has no shim
//@compile-flags: -Zmiri-disable-isolation

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

fn main() {
    test_socket_close();
}

fn test_socket_close() {
    unsafe {
        let sockfd = errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap();
        errno_check(libc::close(sockfd));
    }
}
