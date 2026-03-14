//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation

#[path = "../../utils/libc.rs"]
mod libc_utils;
use std::io::{self, ErrorKind};
use std::mem::MaybeUninit;

use libc_utils::*;

fn main() {
    test_socket_close();
    test_bind_ipv4();
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
    test_bind_ipv4_invalid_addr_len();
    test_bind_ipv6();

    test_listen();

    test_getsockname_ipv4();
    test_getsockname_ipv4_random_port();
    test_getsockname_ipv4_unbound();
    test_getsockname_ipv6();
}

fn test_socket_close() {
    unsafe {
        let sockfd = errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap();
        errno_check(libc::close(sockfd));
    }
}

fn test_bind_ipv4() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::ipv4_sock_addr(net::IPV4_LOCALHOST, 0);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }
}

/// Tests binding after the `SO_REUSEADDR` socket option has been set on the newly created socket.
fn test_bind_ipv4_reuseaddr() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::ipv4_sock_addr(net::IPV4_LOCALHOST, 0);
    setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_REUSEADDR, 1 as libc::c_int).unwrap();
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
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    // Value should be of type `libc::c_int` which has size 4 bytes.
    // By providing a u64 of size 8 bytes we trigger an invalid length error.
    let err = setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_REUSEADDR, 1u64).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    // check that it is the right kind of `InvalidInput`
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
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
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::ipv4_sock_addr(net::IPV4_LOCALHOST, 0);
    setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_NOSIGPIPE, 1 as libc::c_int).unwrap();
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
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    // Value should be of type `libc::c_int` which has size 4 bytes.
    // By providing a u64 of size 8 bytes we trigger an invalid length error.
    let err = setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_NOSIGPIPE, 1u64).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    // check that it is the right kind of `InvalidInput`
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
}

/// Tests binding an IPv4 socket with an IPv4 address but the addrlen argument
/// has the wrong size.
fn test_bind_ipv4_invalid_addr_len() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::ipv4_sock_addr(net::IPV4_LOCALHOST, 0);
    let err = unsafe {
        errno_result(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            // Add 1 to the address to make the size invalid.
            (size_of::<libc::sockaddr_in>() + 1) as libc::socklen_t,
        ))
        .unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    // check that it is the right kind of `InvalidInput`
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
}

fn test_bind_ipv6() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET6, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::ipv6_sock_addr(net::IPV6_LOCALHOST, 0);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in6).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in6>() as libc::socklen_t,
        ));
    }
}

fn test_listen() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::ipv4_sock_addr(net::IPV4_LOCALHOST, 0);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }

    // Use the supported backlog value to avoid the warning.
    let backlog = 128;

    unsafe {
        errno_check(libc::listen(sockfd, backlog));
    }
}

/// Test the `getsockname` syscall on an IPv4 socket which is bound.
/// The `getsockname` syscall should return the same address as to
/// which the socket was bound to.
fn test_getsockname_ipv4() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::ipv4_sock_addr(net::IPV4_LOCALHOST, 6789);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }
    // Use the supported backlog value to avoid the warning.
    let backlog = 128;

    unsafe {
        errno_check(libc::listen(sockfd, backlog));
    }

    let sockname =
        sockname(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) }).unwrap();

    let LibcSocketAddr::V4(sock_addr) = sockname else {
        // We bound an IPv4 address so we also expect
        // an IPv4 address to be returned.
        panic!()
    };

    assert_eq!(addr.sin_family, sock_addr.sin_family);
    assert_eq!(addr.sin_port, sock_addr.sin_port);
    assert_eq!(addr.sin_addr.s_addr, sock_addr.sin_addr.s_addr);
}

/// Test the `getsockname` syscall on an IPv4 socket which is bound
/// but the port was zero.
/// The `getsockname` syscall should return the same address as to
/// which the socket was bound to but the port should be non-zero.
fn test_getsockname_ipv4_random_port() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    // Use zero-port to let the OS choose a free port to bind to.
    let addr = net::ipv4_sock_addr(net::IPV4_LOCALHOST, 0);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }
    // Use the supported backlog value to avoid the warning.
    let backlog = 128;

    unsafe {
        errno_check(libc::listen(sockfd, backlog));
    }

    let sockname =
        sockname(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) }).unwrap();

    let LibcSocketAddr::V4(sock_addr) = sockname else {
        // We bound an IPv4 address so we also expect
        // an IPv4 address to be returned.
        panic!()
    };
    assert_eq!(addr.sin_family, sock_addr.sin_family);
    // The bound port must not be the zero port.
    assert!(sock_addr.sin_port > 0);
    assert_eq!(addr.sin_addr.s_addr, sock_addr.sin_addr.s_addr);
}

/// Test the `getsockname` syscall on an IPv4 socket which is not bound.
/// The `getsockname` syscall should return 0.0.0.0:0
fn test_getsockname_ipv4_unbound() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    let sockname =
        sockname(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) }).unwrap();

    // Libc representation of an unspecified IPv4 address with zero port.
    let addr = net::ipv4_sock_addr([0, 0, 0, 0], 0);
    let LibcSocketAddr::V4(sock_addr) = sockname else {
        // We bound an IPv4 address so we also expect
        // an IPv4 address to be returned.
        panic!()
    };

    assert_eq!(addr.sin_family, sock_addr.sin_family);
    assert_eq!(addr.sin_port, sock_addr.sin_port);
    assert_eq!(addr.sin_addr.s_addr, sock_addr.sin_addr.s_addr);
}

/// Test the `getsockname` syscall on an IPv6 socket which is bound.
/// The `getsockname` syscall should return the same address as to
/// which the socket was bound to.
fn test_getsockname_ipv6() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET6, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::ipv6_sock_addr(net::IPV6_LOCALHOST, 1234);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in6).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in6>() as libc::socklen_t,
        ));
    }
    // Use the supported backlog value to avoid the warning.
    let backlog = 128;

    unsafe {
        errno_check(libc::listen(sockfd, backlog));
    }

    let sockname =
        sockname(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) }).unwrap();

    let LibcSocketAddr::V6(sock_addr) = sockname else {
        // We bound an IPv6 address so we also expect
        // an IPv6 address to be returned.
        panic!()
    };

    assert_eq!(addr.sin6_family, sock_addr.sin6_family);
    assert_eq!(addr.sin6_port, sock_addr.sin6_port);
    assert_eq!(addr.sin6_flowinfo, sock_addr.sin6_flowinfo);
    assert_eq!(addr.sin6_scope_id, sock_addr.sin6_scope_id);
    assert_eq!(addr.sin6_addr.s6_addr, sock_addr.sin6_addr.s6_addr);
}

/// Set a socket option. It's the caller's responsibility to ensure that `T` is
/// associated with the given socket option.
///
/// This function is directly copied from the standard library implementation
/// for sockets on UNIX targets.
fn setsockopt<T>(
    sockfd: i32,
    level: libc::c_int,
    option_name: libc::c_int,
    option_value: T,
) -> io::Result<()> {
    let option_len = size_of::<T>() as libc::socklen_t;

    errno_result(unsafe {
        libc::setsockopt(
            sockfd,
            level,
            option_name,
            (&raw const option_value) as *const _,
            option_len,
        )
    })?;
    Ok(())
}

enum LibcSocketAddr {
    V4(libc::sockaddr_in),
    V6(libc::sockaddr_in6),
}

/// Wraps a call to a platform function that returns a socket address.
/// This is very much the same as the function with the same name in the
/// standard library implementation.
fn sockname<F>(f: F) -> io::Result<LibcSocketAddr>
where
    F: FnOnce(*mut libc::sockaddr, *mut libc::socklen_t) -> libc::c_int,
{
    let mut storage = MaybeUninit::<libc::sockaddr_storage>::zeroed();
    let mut len = size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    errno_result(f(storage.as_mut_ptr().cast(), &mut len))?;
    // SAFETY:
    // The caller guarantees that the storage has been successfully initialized
    // and its size written to `len` if `f` returns a success.
    unsafe {
        match (*storage.as_ptr()).ss_family as libc::c_int {
            libc::AF_INET => {
                assert!(len as usize >= size_of::<libc::sockaddr_in>());
                Ok(LibcSocketAddr::V4(*(storage.as_ptr() as *const _ as *const libc::sockaddr_in)))
            }
            libc::AF_INET6 => {
                assert!(len as usize >= size_of::<libc::sockaddr_in6>());
                Ok(LibcSocketAddr::V6(*(storage.as_ptr() as *const _ as *const libc::sockaddr_in6)))
            }
            _ => Err(io::Error::new(ErrorKind::InvalidInput, "invalid argument")),
        }
    }
}
