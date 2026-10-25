//@only-target: linux android illumos
//@compile-flags: -Zmiri-disable-isolation
//@run-native

#[path = "../../utils/libc.rs"]
mod libc_utils;

use libc_utils::epoll::*;
use libc_utils::*;

fn main() {
    test_epoll_udp_socket();
}

/// Test creating a non-blocking UDP socket and adding it to an epoll instance.
/// The socket should be writable.
fn test_epoll_udp_socket() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    unsafe {
        // Change socket to be non-blocking.
        errno_check(libc::fcntl(sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Add socket with READABLE and WRITABLE interest to epoll.
    epoll_ctl_add(epfd, sockfd, EPOLLIN | EPOLLOUT | EPOLLET | EPOLLERR).unwrap();

    // The socket should always be writable.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: sockfd }]);
}
