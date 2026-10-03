//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation
//@revisions: tcp udp
//@run-native

#[path = "../../utils/libc.rs"]
mod libc_utils;
#[path = "../../utils/mod.rs"]
mod utils;

use libc_utils::*;

// This file contains tests which are executed with different socket types (e.g., TCP, UDP, ...).
// Every socket type has its own revision and based on that the `SOCKET_TYPE` and `SOCKET_PROTOCOL`
// constants get set.

cfg_select! {
    tcp => {
        const SOCKET_TYPE: libc::c_int = libc::SOCK_STREAM;
        const SOCKET_PROTOCOL: libc::c_int = libc::IPPROTO_TCP;
    }
    udp => {
        const SOCKET_TYPE: libc::c_int = libc::SOCK_DGRAM;
        const SOCKET_PROTOCOL: libc::c_int = libc::IPPROTO_UDP;
    }
}

fn main() {
    test_create_close();
    test_create_close_with_protocol();
}

/// Test creating a socket and then closing it afterwards.
fn test_create_close() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };

    let flags = unsafe { errno_result(libc::fcntl(sockfd, libc::F_GETFL, 0)).unwrap() };

    // Ensure that socket is initially blocking.
    assert_eq!(flags & libc::O_NONBLOCK, 0);

    unsafe { errno_check(libc::close(sockfd)) };
}

/// Test creating a socket and then closing it afterwards but we explicitly
/// specify which protocol the socket should use.
fn test_create_close_with_protocol() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, SOCKET_PROTOCOL)).unwrap() };

    let flags = unsafe { errno_result(libc::fcntl(sockfd, libc::F_GETFL, 0)).unwrap() };

    // Ensure that socket is initially blocking.
    assert_eq!(flags & libc::O_NONBLOCK, 0);

    unsafe { errno_check(libc::close(sockfd)) };
}
