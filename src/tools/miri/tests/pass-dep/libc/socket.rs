//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation
//@revisions: tcp udp
//@run-native

#[path = "../../utils/libc.rs"]
mod libc_utils;
#[path = "../../utils/mod.rs"]
mod utils;

use std::slice;

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

    test_getsockopt_truncate();

    test_getsockname_ipv4();
    test_getsockname_ipv4_random_port();
    test_getsockname_ipv4_unbound();
    test_getsockname_ipv6();
    test_getsockname_ipv6_unbound();
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

/// Test that the value gets silently truncated when a too small
/// length is provided and that the length gets reduced when the value
/// is smaller than the provided length.
fn test_getsockopt_truncate() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };

    // Set the read timeout for the socket.
    // We use a multiple of 4ms for the `usec` since Linux seems to do rounding.
    let new_timeout = libc::timeval { tv_sec: 123, tv_usec: 40_000 };
    net::setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_RCVTIMEO, new_timeout).unwrap();

    let mut option_value = std::mem::MaybeUninit::<[u8; 5]>::zeroed();
    // The actual `timeval` length is more than 5 bytes.
    let mut short_option_len = 5 as libc::socklen_t;
    assert!(short_option_len < size_of::<libc::timeval>() as libc::socklen_t);

    let timeout =
        net::getsockopt::<libc::timeval>(sockfd, libc::SOL_SOCKET, libc::SO_RCVTIMEO).unwrap();
    // Ensure that we get the same value back as we just set.
    assert_eq!(timeout.tv_sec, new_timeout.tv_sec);
    assert_eq!(timeout.tv_usec, new_timeout.tv_usec);

    errno_result(unsafe {
        libc::getsockopt(
            sockfd,
            libc::SOL_SOCKET,
            libc::SO_RCVTIMEO,
            option_value.as_mut_ptr().cast(),
            &mut short_option_len,
        )
    })
    .unwrap();
    // Ensure that the size wasn't changed.
    assert_eq!(short_option_len, 5);
    let truncated_timeout = unsafe { option_value.assume_init() };

    unsafe {
        let timeout_ptr = (&new_timeout) as *const libc::timeval as *const u8;
        // Assert that the value was silently truncated.
        assert_eq!(&truncated_timeout, slice::from_raw_parts(timeout_ptr, 5));
    }

    let mut option_value = std::mem::MaybeUninit::<libc::timeval>::zeroed();
    // The actual length is smaller than this.
    let mut long_option_len = (size_of::<libc::timeval>() + 2) as libc::socklen_t;

    errno_result(unsafe {
        libc::getsockopt(
            sockfd,
            libc::SOL_SOCKET,
            libc::SO_RCVTIMEO,
            option_value.as_mut_ptr().cast(),
            &mut long_option_len,
        )
    })
    .unwrap();
    // Ensure that the size was shortened to the actual length.
    assert_eq!(long_option_len, size_of::<libc::timeval>() as libc::socklen_t);
    // The returned timeout should be the same value as we just set.
    let untruncated_timeout = unsafe { option_value.assume_init() };
    assert_eq!(untruncated_timeout.tv_sec, new_timeout.tv_sec);
    assert_eq!(untruncated_timeout.tv_usec, new_timeout.tv_usec);
}

/// Test the `getsockname` syscall on an IPv4 socket which is bound.
/// The `getsockname` syscall should return the same address as to
/// which the socket was bound to.
fn test_getsockname_ipv4() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };
    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 6789);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }

    let (_, sock_addr) =
        net::sockname_ipv4(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) })
            .unwrap();

    assert_eq!(addr.sin_family, sock_addr.sin_family);
    assert_eq!(addr.sin_port, sock_addr.sin_port);
    assert_eq!(addr.sin_addr.s_addr, sock_addr.sin_addr.s_addr);
}

/// Test the `getsockname` syscall on an IPv4 socket which is bound
/// but the port was zero.
/// The `getsockname` syscall should return the same address as to
/// which the socket was bound to but the port should be non-zero.
fn test_getsockname_ipv4_random_port() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };
    // Use zero-port to let the OS choose a free port to bind to.
    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 0);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }

    let (_, sock_addr) =
        net::sockname_ipv4(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) })
            .unwrap();

    assert_eq!(addr.sin_family, sock_addr.sin_family);
    // The bound port must not be the zero port.
    assert!(sock_addr.sin_port > 0);
    assert_eq!(addr.sin_addr.s_addr, sock_addr.sin_addr.s_addr);
}

/// Test the `getsockname` syscall on an IPv4 socket which is not bound.
/// The `getsockname` syscall should return 0.0.0.0:0
fn test_getsockname_ipv4_unbound() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, SOCKET_TYPE, 0)).unwrap() };

    let (_, sock_addr) =
        net::sockname_ipv4(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) })
            .unwrap();

    // Libc representation of an unspecified IPv4 address with zero port.
    let addr = net::sock_addr_ipv4([0, 0, 0, 0], 0);

    assert_eq!(addr.sin_family, sock_addr.sin_family);
    assert_eq!(addr.sin_port, sock_addr.sin_port);
    assert_eq!(addr.sin_addr.s_addr, sock_addr.sin_addr.s_addr);
}

/// Test the `getsockname` syscall on an IPv6 socket which is bound.
/// The `getsockname` syscall should return the same address as to
/// which the socket was bound to.
fn test_getsockname_ipv6() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET6, SOCKET_TYPE, 0)).unwrap() };
    let addr = net::sock_addr_ipv6(net::IPV6_LOCALHOST, 1234);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in6).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in6>() as libc::socklen_t,
        ));
    }

    let (_, sock_addr) =
        net::sockname_ipv6(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) })
            .unwrap();

    assert_eq!(addr.sin6_family, sock_addr.sin6_family);
    assert_eq!(addr.sin6_port, sock_addr.sin6_port);
    assert_eq!(addr.sin6_flowinfo, sock_addr.sin6_flowinfo);
    assert_eq!(addr.sin6_scope_id, sock_addr.sin6_scope_id);
    assert_eq!(addr.sin6_addr.s6_addr, sock_addr.sin6_addr.s6_addr);
}

/// Test the `getsockname` syscall on an IPv6 socket which is not bound.
/// The `getsockname` syscall should return [::]:0 with flowinfo and scope id
/// of zero.
fn test_getsockname_ipv6_unbound() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET6, SOCKET_TYPE, 0)).unwrap() };

    let (_, sock_addr) =
        net::sockname_ipv6(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) })
            .unwrap();

    // Libc representation of an unspecified IPv6 address with zero port, flowinfo and scope id.
    let addr = net::sock_addr_full_ipv6(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        /* port */ 0,
        /* flowinfo */ 0,
        /* scope_id */ 0,
    );

    assert_eq!(addr.sin6_family, sock_addr.sin6_family);
    assert_eq!(addr.sin6_port, sock_addr.sin6_port);
    assert_eq!(addr.sin6_flowinfo, sock_addr.sin6_flowinfo);
    assert_eq!(addr.sin6_scope_id, sock_addr.sin6_scope_id);
    assert_eq!(addr.sin6_addr.s6_addr, sock_addr.sin6_addr.s6_addr);
}
