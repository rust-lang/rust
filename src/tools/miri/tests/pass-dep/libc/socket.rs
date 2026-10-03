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

    test_bind_ipv4();
    test_bind_ipv4_invalid_addr_len();
    test_bind_ipv4_reuseaddr();
    test_set_reuseaddr_invalid_len();
    #[cfg(any(
        target_os = "macos",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "dragonfly"
    ))]
    {
        test_bind_ipv4_nosigpipe();
        test_set_nosigpipe_invalid_len();
    }
    test_bind_ipv6();
    test_bind_wrong_address_family();
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

/// Test binding a newly created socket to an IPv4 address.
fn test_bind_ipv4() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };
    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 0);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }
}

/// Tests binding an IPv4 socket with an IPv4 address but the addrlen argument
/// has the wrong size.
fn test_bind_ipv4_invalid_addr_len() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };
    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 0);
    // A too small size is invalid.
    let err = unsafe {
        errno_result(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            (size_of::<libc::sockaddr_in>() - 1) as libc::socklen_t,
        ))
        .unwrap_err()
    };
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));

    if cfg!(miri) {
        // A too big size is also invalid. Some native hosts (e.g. Linux)
        // allow too big sizes, so we skip this test on native hosts:
        // <https://github.com/rust-lang/miri/pull/5059#discussion_r3305439221>
        let err = unsafe {
            errno_result(libc::bind(
                sockfd,
                (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
                (size_of::<libc::sockaddr_in>() + 1) as libc::socklen_t,
            ))
            .unwrap_err()
        };
        assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
    }
}

/// Tests binding after the `SO_REUSEADDR` socket option has been set on the newly created socket.
fn test_bind_ipv4_reuseaddr() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };
    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 0);
    net::setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_REUSEADDR, 1 as libc::c_int).unwrap();
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }
}

/// Tests setting the `SO_REUSEADDR` socket option but with an invalid length.
fn test_set_reuseaddr_invalid_len() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };
    // Value should be of type `libc::c_int` which has size 4 bytes.
    // By providing an u16 of size 2 bytes we trigger an invalid length error.
    let err = net::setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_REUSEADDR, 1u16).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));

    // By providing an u64 of size 8 bytes the behavior differs between native hosts and Miri.
    let result = net::setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_REUSEADDR, 1u64);
    match result {
        Err(err) => {
            // Check that this is the right error.
            assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
        }
        Ok(_) => {
            // Some native hosts just ignore too large inputs and only look at a prefix.
            // On Miri we require an exact size.
            assert!(!cfg!(miri));
        }
    }
}

#[cfg(any(
    target_os = "macos",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "dragonfly"
))]
/// Tests binding after the `SO_NOSIGPIPE` socket option has been set on the newly created socket.
/// That flag only exists on BSD-like OSes.
fn test_bind_ipv4_nosigpipe() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };
    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 0);
    net::setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_NOSIGPIPE, 1 as libc::c_int).unwrap();
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }
}

#[cfg(any(
    target_os = "macos",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "dragonfly"
))]
/// Tests setting the `SO_NOSIGPIPE` socket option but with an invalid length.
fn test_set_nosigpipe_invalid_len() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };
    // Value should be of type `libc::c_int` which has size 4 bytes.
    // By providing a u16 of size 2 bytes we trigger an invalid length error.
    let err = net::setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_NOSIGPIPE, 1u16).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
}

/// Test binding a newly created socket to an IPv6 address.
fn test_bind_ipv6() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET6, SOCKET_TYPE, 0)).unwrap() };
    let addr = net::sock_addr_ipv6(net::IPV6_LOCALHOST, 0);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in6).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in6>() as libc::socklen_t,
        ));
    }
}

/// Test binding a newly created IPv6 UDP socket to an IPv4 address.
fn test_bind_wrong_address_family() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET6, SOCKET_TYPE, 0)).unwrap() };
    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 0);

    let err = unsafe {
        errno_result(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ))
        .unwrap_err()
    };

    if cfg!(any(target_os = "linux", target_os = "android")) || cfg!(all(udp, target_os = "macos"))
    {
        // For both UDP and TCP sockets Linux and Android targets return EINVAL
        // instead of EAFNOSUPPORT. However, for UDP sockets macOS also returns
        // EINVAL while for TCP sockets macOS returns EAFNOSUPPORT.
        assert_eq!(err.raw_os_error(), Some(libc::EINVAL))
    } else {
        assert_eq!(err.raw_os_error(), Some(libc::EAFNOSUPPORT))
    }
}
