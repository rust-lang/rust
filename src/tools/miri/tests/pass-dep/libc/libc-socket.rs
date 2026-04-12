//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation

#[path = "../../utils/libc.rs"]
mod libc_utils;
#[path = "../../utils/mod.rs"]
mod utils;

use std::io::ErrorKind;
use std::thread;
use std::time::Duration;

use libc_utils::*;
use utils::check_nondet;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

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

    test_accept_connect();
    test_send_peek_recv();
    test_partial_send_recv();
    test_write_read();

    test_getsockname_ipv4();
    test_getsockname_ipv4_random_port();
    test_getsockname_ipv4_unbound();
    test_getsockname_ipv6();

    test_getpeername_ipv4();
    test_getpeername_ipv6();
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
    // By providing a u64 of size 8 bytes we trigger an invalid length error.
    let err = net::setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_REUSEADDR, 1u64).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    // Check that it is the right kind of `InvalidInput`.
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
    // By providing a u64 of size 8 bytes we trigger an invalid length error.
    let err = net::setsockopt(sockfd, libc::SOL_SOCKET, libc::SO_NOSIGPIPE, 1u64).unwrap_err();
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
    // Check that it is the right kind of `InvalidInput`.
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
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
    let (server_sockfd, addr) = net::make_listener_ipv4(0).unwrap();
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

/// Test sending bytes into a connected stream and then peeking and receiving
/// them from the other end.
/// We especially want to test that the peeking doesn't remove the bytes from
/// the queue.
fn test_send_peek_recv() {
    let (server_sockfd, addr) = net::make_listener_ipv4(0).unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();

        // Write the bytes into the stream.
        unsafe {
            errno_result(libc_utils::write_all_generic(
                TEST_BYTES.as_ptr().cast(),
                TEST_BYTES.len(),
                libc_utils::NoRetry,
                |buf, count| libc::send(peerfd, buf, count, 0),
            ))
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
        errno_result(libc_utils::read_all_generic(
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc_utils::NoRetry,
            |buf, count| libc::recv(client_sockfd, buf, count, 0),
        ))
        .unwrap()
    };
    assert_eq!(&buffer, TEST_BYTES);

    server_thread.join().unwrap();
}

/// Test that we actually do partial sends and partial receives for sockets.
fn test_partial_send_recv() {
    let (server_sockfd, addr) = net::make_listener_ipv4(0).unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();

        // Yield back to client to test that we do incomplete writes.
        thread::sleep(Duration::from_millis(10));

        // We know the buffer contains enough bytes to test incomplete reads.

        // Ensure we sometimes do incomplete reads.
        check_nondet(|| {
            let mut buffer = [0u8; 4];
            let bytes_read =
                unsafe { errno_result(libc::read(peerfd, buffer.as_mut_ptr().cast(), 4)).unwrap() };
            bytes_read == 4
        });
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    // Ensure we sometimes do incomplete writes.
    check_nondet(|| {
        let bytes_written =
            unsafe { errno_result(libc::write(client_sockfd, [0; 4].as_ptr().cast(), 4)).unwrap() };
        bytes_written == 4
    });

    let buffer = [0u8; 100_000];
    // Write a lot of bytes into the socket such that we can test
    // incomplete reads.
    unsafe {
        errno_result(libc_utils::write_all(client_sockfd, buffer.as_ptr().cast(), buffer.len()))
            .unwrap()
    };

    server_thread.join().unwrap();
}

/// Test writing bytes into a connected stream and then reading them
/// from the other end.
/// We want to test this because `write` and `read` should be the same as
/// `send` and `recv` with zero flags.
fn test_write_read() {
    let (server_sockfd, addr) = net::make_listener_ipv4(0).unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();

        // Write some bytes into the stream.
        let bytes_written = unsafe {
            errno_result(libc_utils::write_all(
                peerfd,
                TEST_BYTES.as_ptr().cast(),
                TEST_BYTES.len(),
            ))
            .unwrap()
        };
        assert_eq!(bytes_written as usize, TEST_BYTES.len());
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    let mut buffer = [0; TEST_BYTES.len()];
    unsafe {
        errno_result(libc_utils::read_all(client_sockfd, buffer.as_mut_ptr().cast(), buffer.len()))
            .unwrap()
    };
    assert_eq!(&buffer, TEST_BYTES);

    server_thread.join().unwrap();
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
    let (server_sockfd, addr) = net::make_listener_ipv4(0).unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || net::accept_ipv4(server_sockfd).unwrap());

    net::connect_ipv4(client_sockfd, addr).unwrap();

    let (_, peer_addr) = net::sockname_ipv4(|storage, len| unsafe {
        libc::getpeername(client_sockfd, storage, len)
    })
    .unwrap();

    assert_eq!(addr.sin_family, peer_addr.sin_family);
    assert_eq!(addr.sin_port, peer_addr.sin_port);
    assert_eq!(addr.sin_addr.s_addr, peer_addr.sin_addr.s_addr);

    server_thread.join().unwrap();
}

/// Test the `getpeername` syscall on an IPv6 socket.
/// For a connected socket, the `getpeername` syscall should
/// return the same address as the socket was connected to.
fn test_getpeername_ipv6() {
    let (server_sockfd, addr) = net::make_listener_ipv6(0).unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET6, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || net::accept_ipv6(server_sockfd).unwrap());

    net::connect_ipv6(client_sockfd, addr).unwrap();

    let (_, peer_addr) = net::sockname_ipv6(|storage, len| unsafe {
        libc::getpeername(client_sockfd, storage, len)
    })
    .unwrap();

    assert_eq!(addr.sin6_family, peer_addr.sin6_family);
    assert_eq!(addr.sin6_port, peer_addr.sin6_port);
    assert_eq!(addr.sin6_flowinfo, peer_addr.sin6_flowinfo);
    assert_eq!(addr.sin6_scope_id, peer_addr.sin6_scope_id);
    assert_eq!(addr.sin6_addr.s6_addr, peer_addr.sin6_addr.s6_addr);

    server_thread.join().unwrap();
}
