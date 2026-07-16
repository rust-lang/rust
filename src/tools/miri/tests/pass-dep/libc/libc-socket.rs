//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation
//@run-native

#[path = "../../utils/libc.rs"]
mod libc_utils;
#[path = "../../utils/mod.rs"]
mod utils;

use std::io::ErrorKind;
use std::time::{Duration, Instant};
use std::{ptr, slice, thread};

use libc_utils::*;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_create_close();
    test_create_close_tcp();
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

    test_accept_connect();
    test_connect_error();
    test_send_peek_recv();
    test_write_read();
    test_readv();
    test_writev();

    test_getsockname_ipv4();
    test_getsockname_ipv4_random_port();
    test_getsockname_ipv4_unbound();
    test_getsockname_ipv4_connect();
    test_getsockname_ipv6();

    test_getpeername_ipv4();
    test_getpeername_ipv6();

    test_shutdown();
    test_shutdown_readable_after_write_close();
    test_shutdown_writable_after_read_close();

    test_getsockopt_truncate();

    test_sockopt_sndtimeo();
    test_sockopt_rcvtimeo();
}

/// Test creating a socket and then closing it afterwards.
fn test_create_close() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    let flags = unsafe { errno_result(libc::fcntl(sockfd, libc::F_GETFL, 0)).unwrap() };

    // Ensure that socket is initially blocking.
    assert_eq!(flags & libc::O_NONBLOCK, 0);

    unsafe { errno_check(libc::close(sockfd)) };
}

/// Test creating a socket and then closing it afterwards but we explicitly
/// specify that the TCP protocol should be used.
fn test_create_close_tcp() {
    let sockfd = unsafe {
        errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, libc::IPPROTO_TCP)).unwrap()
    };

    let flags = unsafe { errno_result(libc::fcntl(sockfd, libc::F_GETFL, 0)).unwrap() };

    // Ensure that socket is initially blocking.
    assert_eq!(flags & libc::O_NONBLOCK, 0);

    unsafe { errno_check(libc::close(sockfd)) };
}

fn test_bind_ipv4() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 0);
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
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    // Value should be of type `libc::c_int` which has size 4 bytes.
    // By providing an u16 of size 2 bytes we trigger an invalid length error.
    let err = net::setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_REUSEADDR, 1u16).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    // Check that it is the right kind of `InvalidInput`.
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));

    // By providing an u64 of size 8 bytes the behavior differs between native hosts and Miri.
    let result = net::setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_REUSEADDR, 1u64);
    match result {
        Err(err) => {
            // Check that this is the right error.
            assert_eq!(err.kind(), ErrorKind::InvalidInput);
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
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
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
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    // Value should be of type `libc::c_int` which has size 4 bytes.
    // By providing a u16 of size 2 bytes we trigger an invalid length error.
    let err = net::setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_NOSIGPIPE, 1u16).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    // Check that it is the right kind of `InvalidInput`.
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
}

/// Tests binding an IPv4 socket with an IPv4 address but the addrlen argument
/// has the wrong size.
fn test_bind_ipv4_invalid_addr_len() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
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
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    // Check that it is the right kind of `InvalidInput`.
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
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        // Check that it is the right kind of `InvalidInput`.
        assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
    }
}

fn test_bind_ipv6() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET6, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::sock_addr_ipv6(net::IPV6_LOCALHOST, 0);
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
    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 0);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }

    unsafe {
        errno_check(libc::listen(sockfd, 16));
    }
}

/// Test accepting connections by running a server in a separate thread and connecting clients
/// from the main thread.
/// This function tests both
/// - Connecting when the server is already accepting
/// - Accepting when there is already an incoming connection
fn test_accept_connect() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();

        let flags = unsafe { errno_result(libc::fcntl(peerfd, libc::F_GETFL, 0)).unwrap() };

        // Ensure that peer socket is blocking.
        assert_eq!(flags & libc::O_NONBLOCK, 0);

        // Yield back to the client thread to test whether calling `connect` first also
        // works.
        thread::sleep(Duration::from_millis(10));

        net::accept_ipv4(server_sockfd).unwrap();
    });

    // Yield to server thread to ensure `accept` is called before we try
    // to connect.
    thread::sleep(Duration::from_millis(10));

    // Test connecting to an already accepting server.
    net::connect_ipv4(client_sockfd, addr).unwrap();

    // Server thread should now be in its `sleep`.
    // Test connecting when there is no actively ongoing `accept`.
    // We need a new client socket since we cannot connect a socket multiple times.
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    net::connect_ipv4(client_sockfd, addr).unwrap();

    server_thread.join().unwrap();
}

/// Test connecting to an address where nothing is listening and ensure the error matches what
/// the standard library expects.
fn test_connect_error() {
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Connecting to a zero port fails on all host platforms.
    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 0);

    let err = net::connect_ipv4(client_sockfd, addr).unwrap_err();
    // Ensure that we fail for the same reasons as the standard library expects.
    assert!(matches!(
        err.kind(),
        ErrorKind::ConnectionRefused
            | ErrorKind::InvalidInput
            | ErrorKind::AddrInUse
            | ErrorKind::AddrNotAvailable
            | ErrorKind::NetworkUnreachable
    ));
}

/// Test sending bytes into a connected stream and then peeking and receiving
/// them from the other end.
/// We especially want to test that the peeking doesn't remove the bytes from
/// the queue.
fn test_send_peek_recv() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();

        // Write the bytes into the stream.
        unsafe {
            libc_utils::write_all_generic(
                TEST_BYTES.as_ptr().cast(),
                TEST_BYTES.len(),
                libc_utils::NoRetry,
                |buf, count| libc::send(peerfd, buf, count, 0),
            )
            .unwrap()
        };
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    let mut buffer = [0; TEST_BYTES.len()];
    let bytes_read = unsafe {
        errno_result(libc::recv(
            client_sockfd,
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc::MSG_PEEK,
        ))
        .unwrap()
    } as usize;

    // We cannot have "peek all" since peeks don't remove the bytes from the buffer.
    // Thus, we only check that the bytes we read are a prefix of the bytes we wrote.
    assert_eq!(&buffer[..bytes_read], &TEST_BYTES[..bytes_read]);

    // Since the bytes aren't removed from the queue by peeking, we should be
    // able to read the same bytes again into a new buffer.

    let mut buffer = [0; TEST_BYTES.len()];
    unsafe {
        libc_utils::read_exact_generic(
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc_utils::NoRetry,
            |buf, count| libc::recv(client_sockfd, buf, count, 0),
        )
        .unwrap()
    };
    assert_eq!(&buffer, TEST_BYTES);

    server_thread.join().unwrap();
}

/// Test writing bytes into a connected stream and then reading them
/// from the other end.
/// We want to test this because `write` and `read` should be the same as
/// `send` and `recv` with zero flags.
fn test_write_read() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();

        // Write some bytes into the stream.
        libc_utils::write_all(peerfd, TEST_BYTES).unwrap();
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    let mut buffer = [0; TEST_BYTES.len()];
    libc_utils::read_exact(client_sockfd, &mut buffer).unwrap();
    assert_eq!(&buffer, TEST_BYTES);

    server_thread.join().unwrap();
}

/// Test vectored reads with multiple buffers on a connected socket.
fn test_readv() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    net::connect_ipv4(client_sockfd, addr).unwrap();
    let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();

    libc_utils::write_all(peerfd, TEST_BYTES).unwrap();

    let mut buffer = [0u8; TEST_BYTES.len()];
    let (buffer1, buffer2) = buffer.split_at_mut(2);

    let iov = [
        libc::iovec { iov_base: ptr::null_mut::<libc::c_void>(), iov_len: 0 as libc::size_t },
        libc::iovec {
            iov_base: buffer1.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: buffer1.len() as libc::size_t,
        },
        libc::iovec {
            iov_base: buffer2.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: buffer2.len() as libc::size_t,
        },
    ];

    let num = unsafe {
        errno_result(libc::readv(client_sockfd, iov.as_ptr(), iov.len() as libc::c_int)).unwrap()
    };
    assert_eq!(num as usize, TEST_BYTES.len());
    // The vectored read should read the entire buffer because we don't have
    // short reads on sockets.
    assert_eq!(&buffer, TEST_BYTES);
}

/// Test vectored writes with multiple buffers on a connected socket.
fn test_writev() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    net::connect_ipv4(client_sockfd, addr).unwrap();
    let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();

    let mut write_buffer = TEST_BYTES.to_owned();
    let (buffer1, buffer2) = write_buffer.split_at_mut(3);

    let iov = [
        libc::iovec { iov_base: ptr::null_mut::<libc::c_void>(), iov_len: 0 as libc::size_t },
        libc::iovec {
            iov_base: buffer1.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: buffer1.len() as libc::size_t,
        },
        libc::iovec {
            iov_base: buffer2.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: buffer2.len() as libc::size_t,
        },
    ];

    let num = unsafe {
        errno_result(libc::writev(client_sockfd, iov.as_ptr(), iov.len() as libc::c_int)).unwrap()
    };
    assert_eq!(num as usize, TEST_BYTES.len());

    let mut buffer = [0u8; TEST_BYTES.len()];
    libc_utils::read_exact(peerfd, &mut buffer).unwrap();
    // The vectored write should write the entire buffer because we don't have
    // short writes on sockets.
    assert_eq!(&buffer, TEST_BYTES);
}

/// Test the `getsockname` syscall on an IPv4 socket which is bound.
/// The `getsockname` syscall should return the same address as to
/// which the socket was bound to.
fn test_getsockname_ipv4() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 6789);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }

    unsafe {
        errno_check(libc::listen(sockfd, 16));
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
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    // Use zero-port to let the OS choose a free port to bind to.
    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 0);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }

    unsafe {
        errno_check(libc::listen(sockfd, 16));
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
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    let (_, sock_addr) =
        net::sockname_ipv4(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) })
            .unwrap();

    // Libc representation of an unspecified IPv4 address with zero port.
    let addr = net::sock_addr_ipv4([0, 0, 0, 0], 0);

    assert_eq!(addr.sin_family, sock_addr.sin_family);
    assert_eq!(addr.sin_port, sock_addr.sin_port);
    assert_eq!(addr.sin_addr.s_addr, sock_addr.sin_addr.s_addr);
}

/// Test the `getsockname` syscall on a connected IPv4 socket.
fn test_getsockname_ipv4_connect() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    net::connect_ipv4(client_sockfd, addr).unwrap();
    net::accept_ipv4(server_sockfd).unwrap();

    let (_, sock_addr) = net::sockname_ipv4(|storage, len| unsafe {
        libc::getsockname(client_sockfd, storage, len)
    })
    .unwrap();

    // We want to ensure that the local address is not the unspecified address.
    // Because the bound address could be of any local interface, we cannot
    // test for localhost.
    let addr = net::sock_addr_ipv4([0, 0, 0, 0], 0);

    assert_eq!(addr.sin_family, sock_addr.sin_family);
    assert_ne!(addr.sin_addr.s_addr, sock_addr.sin_addr.s_addr);
    assert!(sock_addr.sin_port > 0);
}

/// Test the `getsockname` syscall on an IPv6 socket which is bound.
/// The `getsockname` syscall should return the same address as to
/// which the socket was bound to.
fn test_getsockname_ipv6() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET6, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::sock_addr_ipv6(net::IPV6_LOCALHOST, 1234);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in6).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in6>() as libc::socklen_t,
        ));
    }

    unsafe {
        errno_check(libc::listen(sockfd, 16));
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

/// Test the `getpeername` syscall on an IPv4 socket.
/// For a connected socket, the `getpeername` syscall should
/// return the same address as the socket was connected to.
fn test_getpeername_ipv4() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    net::connect_ipv4(client_sockfd, addr).unwrap();
    net::accept_ipv4(server_sockfd).unwrap();

    let (_, peer_addr) = net::sockname_ipv4(|storage, len| unsafe {
        libc::getpeername(client_sockfd, storage, len)
    })
    .unwrap();

    assert_eq!(addr.sin_family, peer_addr.sin_family);
    assert_eq!(addr.sin_port, peer_addr.sin_port);
    assert_eq!(addr.sin_addr.s_addr, peer_addr.sin_addr.s_addr);
}

/// Test the `getpeername` syscall on an IPv6 socket.
/// For a connected socket, the `getpeername` syscall should
/// return the same address as the socket was connected to.
fn test_getpeername_ipv6() {
    let (server_sockfd, addr) = net::make_listener_ipv6().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET6, libc::SOCK_STREAM, 0)).unwrap() };

    net::connect_ipv6(client_sockfd, addr).unwrap();
    net::accept_ipv6(server_sockfd).unwrap();

    let (_, peer_addr) = net::sockname_ipv6(|storage, len| unsafe {
        libc::getpeername(client_sockfd, storage, len)
    })
    .unwrap();

    assert_eq!(addr.sin6_family, peer_addr.sin6_family);
    assert_eq!(addr.sin6_port, peer_addr.sin6_port);
    assert_eq!(addr.sin6_flowinfo, peer_addr.sin6_flowinfo);
    assert_eq!(addr.sin6_scope_id, peer_addr.sin6_scope_id);
    assert_eq!(addr.sin6_addr.s6_addr, peer_addr.sin6_addr.s6_addr);
}

/// Test shutting down TCP streams.
fn test_shutdown() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || net::accept_ipv4(server_sockfd).unwrap());

    let mut byte = [0u8];

    net::connect_ipv4(client_sockfd, addr).unwrap();
    let client_dup_sockfd = unsafe { libc::dup(client_sockfd) };

    // Closing should prevent reads/writes.
    unsafe {
        libc::shutdown(client_sockfd, libc::SHUT_RDWR);
        let err = errno_result(libc::write(client_sockfd, [0u8].as_ptr().cast(), 1)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::BrokenPipe);
        let bytes_read =
            errno_result(libc::read(client_sockfd, byte.as_mut_ptr().cast(), 1)).unwrap();
        assert_eq!(bytes_read, 0);
    }

    // Closing should affect previously duplicated handles.
    unsafe {
        let err =
            errno_result(libc::write(client_dup_sockfd, [0u8].as_ptr().cast(), 1)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::BrokenPipe);
        let bytes_read =
            errno_result(libc::read(client_dup_sockfd, byte.as_mut_ptr().cast(), 1)).unwrap();
        assert_eq!(bytes_read, 0);
    }

    // Closing should affect newly duplicated handles.
    unsafe {
        let client_dup2_sockfd = libc::dup(client_sockfd);
        let err =
            errno_result(libc::write(client_dup2_sockfd, [0u8].as_ptr().cast(), 1)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::BrokenPipe);
        let bytes_read =
            errno_result(libc::read(client_dup2_sockfd, byte.as_mut_ptr().cast(), 1)).unwrap();
        assert_eq!(bytes_read, 0);
    }

    server_thread.join().unwrap();
}

/// Test that a socket is still readable after the write end has
/// been closed.
fn test_shutdown_readable_after_write_close() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();
        // Write a single byte which should be read later on.
        unsafe { errno_result(libc::write(peerfd, [1u8].as_ptr().cast(), 1)).unwrap() };
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    unsafe {
        // Close the write end.
        libc::shutdown(client_sockfd, libc::SHUT_WR);
        // Ensure that we're still readable.
        let mut byte = [0u8];
        errno_result(libc::read(client_sockfd, byte.as_mut_ptr().cast(), 1)).unwrap();
        assert_eq!(&byte, &[1u8]);
    }

    server_thread.join().unwrap();
}

/// Test that a socket is still writable after the read end has
/// been closed.
fn test_shutdown_writable_after_read_close() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || net::accept_ipv4(server_sockfd).unwrap());

    net::connect_ipv4(client_sockfd, addr).unwrap();

    unsafe {
        // Close the read end.
        libc::shutdown(client_sockfd, libc::SHUT_RD);
        // Ensure that we're still writable.
        errno_result(libc::write(client_sockfd, [1u8].as_ptr().cast(), 1)).unwrap();
    }

    server_thread.join().unwrap();
}

/// Test that the value gets silently truncated when a too small
/// length is provided and that the length gets reduced when the value
/// is smaller than the provided length.
fn test_getsockopt_truncate() {
    let (sockfd, _) = net::make_listener_ipv4().unwrap();

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

/// Test setting and getting the SO_SNDTIMEO socket option.
/// Also test that writes don't block indefinitely when we
/// have a nonzero timeout.
fn test_sockopt_sndtimeo() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    net::connect_ipv4(client_sockfd, addr).unwrap();
    net::accept_ipv4(server_sockfd).unwrap();

    let timeout =
        net::getsockopt::<libc::timeval>(client_sockfd, libc::SOL_SOCKET, libc::SO_SNDTIMEO)
            .unwrap();
    // By default, no write timeout should be set.
    assert_eq!(timeout.tv_sec, 0);
    assert_eq!(timeout.tv_usec, 0);

    // A 40 millisecond timeout.
    let short_timeout = libc::timeval { tv_sec: 0, tv_usec: 40_000 };
    net::setsockopt(client_sockfd, libc::SOL_SOCKET, libc::SO_SNDTIMEO, short_timeout).unwrap();

    let timeout =
        net::getsockopt::<libc::timeval>(client_sockfd, libc::SOL_SOCKET, libc::SO_SNDTIMEO)
            .unwrap();
    // We should now read the same value as we wrote above.
    assert_eq!(timeout.tv_sec, short_timeout.tv_sec);
    assert_eq!(timeout.tv_usec, short_timeout.tv_usec);

    let buffer = [1u8; 32_000];
    loop {
        let before = Instant::now();
        let result = unsafe {
            errno_result(libc::write(client_sockfd, buffer.as_ptr().cast(), buffer.len()))
        };
        match result {
            Ok(_) => { /* continue to fill up buffer */ }
            // When we get an EAGAIN/EWOULDBLOCK when writing into a blocking socket, we know
            // it's because of the write timeout exceeding because the write buffer
            // is full.
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                // The last write should return an EAGAIN/EWOULDBLOCK after ~40ms instead
                // of blocking indefinitely.
                assert!(before.elapsed() >= Duration::from_millis(40));
                break;
            }
            Err(err) => panic!("unexpected error whilst filling up buffer: {err}"),
        }
    }
}

/// Test setting and getting the SO_RCVTIMEO socket option.
/// Also test that reads don't block indefinitely when we
/// have a nonzero timeout.
fn test_sockopt_rcvtimeo() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    net::connect_ipv4(client_sockfd, addr).unwrap();
    net::accept_ipv4(server_sockfd).unwrap();

    let timeout =
        net::getsockopt::<libc::timeval>(client_sockfd, libc::SOL_SOCKET, libc::SO_RCVTIMEO)
            .unwrap();
    // By default, no read timeout should be set.
    assert_eq!(timeout.tv_sec, 0);
    assert_eq!(timeout.tv_usec, 0);

    // A 40 millisecond timeout.
    let short_timeout = libc::timeval { tv_sec: 0, tv_usec: 40_000 };
    net::setsockopt(client_sockfd, libc::SOL_SOCKET, libc::SO_RCVTIMEO, short_timeout).unwrap();

    let timeout =
        net::getsockopt::<libc::timeval>(client_sockfd, libc::SOL_SOCKET, libc::SO_RCVTIMEO)
            .unwrap();
    // We should now read the same value as we wrote above.
    assert_eq!(timeout.tv_sec, short_timeout.tv_sec);
    assert_eq!(timeout.tv_usec, short_timeout.tv_usec);

    let mut buffer = [0u8; 16];
    // The read should return an EAGAIN/EWOULDBLOCK after ~10ms instead of blocking indefinitely.
    let before = Instant::now();
    let err = unsafe {
        errno_result(libc::read(client_sockfd, buffer.as_mut_ptr().cast(), buffer.len()))
            .unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock);
    // Ensure that we blocked for at least 40 milliseconds.
    assert!(before.elapsed() >= Duration::from_millis(40))
}
