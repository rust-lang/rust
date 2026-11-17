//@only-target: linux android illumos
//@compile-flags: -Zmiri-disable-isolation
//@run-native

#[path = "../../utils/libc.rs"]
mod libc_utils;

use std::io::ErrorKind;

use libc_utils::epoll::*;
use libc_utils::*;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_epoll_udp_socket();
    test_recv_nonblock();
    test_initial_readiness();
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

/// Test receiving bytes from a UDP socket without blocking.
/// Instead of busy waiting until we no longer receive EWOULDBLOCK when trying to
/// read from the client, we register the client socket to epoll and wait for
/// READABLE events.
fn test_recv_nonblock() {
    let (rx_sockfd, addr) = net::make_local_ipv4(libc::SOCK_DGRAM).unwrap();
    let tx_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    unsafe {
        // Change both sockets to be non-blocking.
        errno_check(libc::fcntl(rx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
        errno_check(libc::fcntl(tx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Set the default destination address of the transmitting socket.
    net::connect_ipv4(tx_sockfd, addr).unwrap();

    // Add the receiving socket with a READABLE interest to epoll.
    epoll_ctl_add(epfd, rx_sockfd, EPOLLIN | EPOLLET | EPOLLERR).unwrap();

    // Write some data into the transmitting socket.
    unsafe {
        errno_result(libc::write(tx_sockfd, TEST_BYTES.as_ptr().cast(), TEST_BYTES.len())).unwrap();
    }

    // Ensure that the EPOLLIN readiness is set on the receiving socket.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLIN, data: rx_sockfd }]);

    let mut buffer = [0; TEST_BYTES.len()];

    unsafe {
        errno_result(libc::read(rx_sockfd, buffer.as_mut_ptr().cast(), buffer.len())).unwrap();
    }
    assert_eq!(&buffer, TEST_BYTES);

    // There should be no data left to read in the socket; a read should thus
    // fail with EWOULDBLOCK.
    let err = unsafe {
        errno_result(libc::read(rx_sockfd, buffer.as_mut_ptr().cast(), buffer.len())).unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Ensure that the EPOLLIN readiness is no longer set on the receiving
    // socket since we just read everything from it.
    assert_eq!(current_epoll_readiness(rx_sockfd, EPOLLIN | EPOLLET), 0);
}

/// Test that a socket reports the EPOLLOUT readiness directly after being created.
fn test_initial_readiness() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)) }.unwrap();
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    epoll_ctl_add(epfd, sockfd, EPOLLOUT | EPOLLIN | EPOLLHUP | EPOLLRDHUP).unwrap();

    check_epoll_wait(epfd, &[Ev { events: EPOLLOUT, data: sockfd }], -1);
}
