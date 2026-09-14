//@only-target: linux android illumos
//@compile-flags: -Zmiri-disable-isolation

#![feature(io_error_inprogress)]

#[path = "../../utils/libc.rs"]
mod libc_utils;

use std::io::ErrorKind;

use libc_utils::epoll::*;
use libc_utils::*;

/// Test calling `connect` after the first `connect` failed.
fn main() {
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    unsafe {
        // Change client socket to be non-blocking.
        errno_check(libc::fcntl(client_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // We cannot attempt to connect to a localhost address because
    // it could be the case that a socket from another test is
    // currently listening on `localhost:12321` because we bind to
    // random ports everywhere. For `127.0.1.1` we know that it's a loopback
    // address and thus exists but because it's not the standard loopback
    // address we also assume that nothing is bound to it.
    // The port `12321` is just a random non-zero port because Windows
    // and Apple hosts return EADDRNOTAVAIL when attempting to connect to
    // a zero port.
    let addr = net::sock_addr_ipv4([127, 0, 1, 1], 12321);

    // Non-blocking connect should fail with EINPROGRESS.
    let err = net::connect_ipv4(client_sockfd, addr).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InProgress);

    // Add interest for client socket.
    epoll_ctl_add(epfd, client_sockfd, EPOLLOUT | EPOLLET | libc::EPOLLERR).unwrap();

    // Wait until the socket has an error.
    check_epoll_wait(
        epfd,
        &[Ev { events: libc::EPOLLERR | EPOLLOUT | EPOLLHUP, data: client_sockfd }],
        -1,
    );

    let errno =
        net::getsockopt::<libc::c_int>(client_sockfd, libc::SOL_SOCKET, libc::SO_ERROR).unwrap();
    // Depending on the host we receive different error kinds. Thus, we only check
    // that it's a nonzero error code.
    assert!(errno != 0);

    // Ensure that error readiness is cleared after reading SO_ERROR.
    let readiness = current_epoll_readiness(client_sockfd, EPOLLET | EPOLLOUT | EPOLLERR);
    assert!(readiness & EPOLLERR == 0);

    unsafe {
        //~vERROR: sockets are in an unspecified state after a failed `connect`
        libc::connect(
            client_sockfd,
            (&addr as *const libc::sockaddr_in).cast(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        )
    };
}
