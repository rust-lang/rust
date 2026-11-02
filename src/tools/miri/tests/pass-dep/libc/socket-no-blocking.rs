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
// Every socket type has its own revision and based on that the `SOCKET_TYPE` constant gets set.

cfg_select! {
    tcp => {
        const SOCKET_TYPE: libc::c_int = libc::SOCK_STREAM;
    }
    udp => {
        const SOCKET_TYPE: libc::c_int = libc::SOCK_DGRAM;
    }
}

fn main() {
    test_fcntl_nonblock_opt();
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "solaris",
        target_os = "illumos"
    ))]
    test_sock_nonblock_opt();
    #[cfg(not(any(target_os = "solaris", target_os = "illumos")))]
    test_ioctl_fionbio_op();
}

/// Test that setting the O_NONBLOCK flag changes the blocking state of a socket.
fn test_fcntl_nonblock_opt() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };

    unsafe {
        // Change socket to be non-blocking.
        errno_check(libc::fcntl(sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    let flags = unsafe { errno_result(libc::fcntl(sockfd, libc::F_GETFL, 0)).unwrap() };
    // Ensure that socket is really non-blocking.
    assert_eq!(flags & libc::O_NONBLOCK, libc::O_NONBLOCK);

    unsafe {
        // Change socket back to be blocking.
        errno_check(libc::fcntl(sockfd, libc::F_SETFL, 0));
    }

    let flags = unsafe { errno_result(libc::fcntl(sockfd, libc::F_GETFL, 0)).unwrap() };
    // Ensure that socket is really blocking.
    assert_eq!(flags & libc::O_NONBLOCK, 0);
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "solaris",
    target_os = "illumos"
))]
/// Test creating a non-blocking socket by using the SOCK_NONBLOCK option
/// for the `socket` syscall.
fn test_sock_nonblock_opt() {
    let sockfd = unsafe {
        errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE | libc::SOCK_NONBLOCK, 0)).unwrap()
    };

    let flags = unsafe { errno_result(libc::fcntl(sockfd, libc::F_GETFL, 0)).unwrap() };
    // Ensure that socket is really non-blocking.
    assert_eq!(flags & libc::O_NONBLOCK, libc::O_NONBLOCK);
}

#[cfg(not(any(target_os = "solaris", target_os = "illumos")))]
/// Test changing the blocking state of a socket using the `ioctl(fd, FIONBIO, ...)`
/// syscall.
fn test_ioctl_fionbio_op() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };

    unsafe {
        // Change socket to be non-blocking.
        let mut value = 1 as libc::c_int;
        errno_check(libc::ioctl(sockfd, libc::FIONBIO, &mut value));
    }

    let flags = unsafe { errno_result(libc::fcntl(sockfd, libc::F_GETFL, 0)).unwrap() };
    // Ensure that socket is really non-blocking.
    assert_eq!(flags & libc::O_NONBLOCK, libc::O_NONBLOCK);

    unsafe {
        // Change socket back to be blocking.
        let mut value = 0 as libc::c_int;
        errno_check(libc::ioctl(sockfd, libc::FIONBIO, &mut value));
    }

    let flags = unsafe { errno_result(libc::fcntl(sockfd, libc::F_GETFL, 0)).unwrap() };
    // Ensure that socket is really blocking.
    assert_eq!(flags & libc::O_NONBLOCK, 0);
}
