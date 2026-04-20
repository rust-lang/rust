//@ignore-target: windows
//@compile-flags: -Zmiri-disable-isolation
//@revisions: windows_host unix_host
//@[unix_host] ignore-host: windows
//@[windows_host] only-host: windows

#![feature(io_error_inprogress)]

#[path = "../../utils/libc.rs"]
mod libc_utils;

use std::io::ErrorKind;
use std::thread;
use std::time::Duration;

use libc_utils::*;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

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

    test_accept_nonblock();
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "solaris",
        target_os = "illumos"
    ))]
    test_accept4_sock_nonblock_opt();
    test_connect_nonblock();
    test_send_recv_nonblock();
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "solaris",
        target_os = "illumos"
    ))]
    test_send_recv_dontwait();
    test_write_read_nonblock();

    test_getpeername_ipv4_nonblock();
    test_getpeername_ipv4_nonblock_no_peer();
}

/// Test that setting the O_NONBLOCK flag changes the blocking state of a socket.
fn test_fcntl_nonblock_opt() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

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
        errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM | libc::SOCK_NONBLOCK, 0))
            .unwrap()
    };

    let flags = unsafe { errno_result(libc::fcntl(sockfd, libc::F_GETFL, 0)).unwrap() };
    // Ensure that socket is really non-blocking.
    assert_eq!(flags & libc::O_NONBLOCK, libc::O_NONBLOCK);
}

#[cfg(not(any(target_os = "solaris", target_os = "illumos")))]
/// Test changing the blocking state of a socket using the `ioctl(fd, FIONBIO, ...)`
/// syscall.
fn test_ioctl_fionbio_op() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

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

/// Test that nonblocking TCP server sockets return [`ErrorKind::WouldBlock`] when trying
/// to accept when no incoming connection exists. This also tests that nonblocking server sockets
/// are still able to accept incoming connections should they already exist before the `accept` or
/// `accept4` syscall is called.
fn test_accept_nonblock() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    unsafe {
        // Change server socket to be non-blocking.
        errno_check(libc::fcntl(server_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // This should fail as we don't have an incoming connection for this address.
    let err = net::accept_ipv4(server_sockfd).unwrap_err();
    // Assert that either EAGAIN or EWOULDBLOCK was returned.
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        // Instantly yield to main thread to ensure that the `connect` syscall
        // was called before we call the `accept` on the server.
        thread::sleep(Duration::from_millis(10));

        net::accept_ipv4(server_sockfd).unwrap();
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    server_thread.join().unwrap();
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "solaris",
    target_os = "illumos"
))]
/// Test that calling `accept4` with the SOCK_NONBLOCK flag produces
/// a non-blocking peer socket.
fn test_accept4_sock_nonblock_opt() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::sockname_ipv4(|storage, len| unsafe {
            libc::accept4(server_sockfd, storage, len, libc::SOCK_NONBLOCK)
        })
        .unwrap();

        let flags = unsafe { errno_result(libc::fcntl(peerfd, libc::F_GETFL, 0)).unwrap() };

        // Ensure that peer socket is non-blocking.
        assert_eq!(flags & libc::O_NONBLOCK, libc::O_NONBLOCK);

        let mut buffer = [0u8; 8];
        // Reading from a socket should return EWOULDBLOCK when there is no
        // data written into it.
        let err = unsafe {
            errno_result(libc::read(peerfd, buffer.as_mut_ptr().cast(), buffer.len())).unwrap_err()
        };
        assert_eq!(err.kind(), ErrorKind::WouldBlock);
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    server_thread.join().unwrap();
}

/// Test that connecting to a server socket works when the client
/// socket is non-blocking before the `connect` call.
fn test_connect_nonblock() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    unsafe {
        // Change client socket to be non-blocking.
        errno_check(libc::fcntl(client_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        net::accept_ipv4(server_sockfd).unwrap();
    });

    // Yield to server thread to ensure that it's currently accepting.
    thread::sleep(Duration::from_millis(10));

    // Non-blocking connects always "fail" with EINPROGRESS.
    let err = net::connect_ipv4(client_sockfd, addr).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InProgress);

    loop {
        let result = net::sockname_ipv4(|storage, len| unsafe {
            libc::getpeername(client_sockfd, storage, len)
        });
        match result {
            Ok(_) => {
                // The client is now connected.
                break;
            }
            Err(err) if err.kind() == ErrorKind::NotConnected => {
                // The client is still connecting.
                thread::sleep(Duration::from_millis(10));
            }
            Err(err) => panic!("unexpected error whilst ensuring connection: {err}"),
        }
    }

    server_thread.join().unwrap();
}

/// Test sending bytes into and receiving bytes from a connected stream without blocking.
fn test_send_recv_nonblock() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();
        // `peerfd` is a blocking socket now. But that's okay, the client still does non-blocking
        // reads/writes.

        // Yield back to client so that it starts receiving before we start sending.
        thread::sleep(Duration::from_millis(10));

        unsafe {
            errno_result(libc_utils::write_all_generic(
                TEST_BYTES.as_ptr().cast(),
                TEST_BYTES.len(),
                libc_utils::NoRetry,
                |buf, count| libc::send(peerfd, buf, count, 0),
            ))
            .unwrap()
        };

        // The buffer should contain `TEST_BYTES` at the beginning.
        // This will block until the client sent us this data.
        let mut buffer = [0; TEST_BYTES.len()];
        unsafe {
            errno_result(libc_utils::read_all_generic(
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                libc_utils::NoRetry,
                |buf, count| libc::recv(peerfd, buf, count, 0),
            ))
            .unwrap()
        };
        assert_eq!(&buffer, TEST_BYTES);
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    unsafe {
        // Change client socket to be non-blocking.
        errno_check(libc::fcntl(client_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // We are connected and the server socket is not writing.

    let mut buffer = [0; TEST_BYTES.len()];
    // Receiving from a socket when the peer is not writing is
    // not possible without blocking.
    let err = unsafe {
        errno_result(libc::recv(client_sockfd, buffer.as_mut_ptr().cast(), buffer.len(), 0))
            .unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Try to receive bytes from the peer socket without blocking.
    // Since the peer socket might do partial writes, we might need to
    // sleep multiple times until we received everything.

    unsafe {
        errno_result(libc_utils::read_all_generic(
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc_utils::RetryAfter(Duration::from_millis(10)),
            |buf, count| libc::recv(client_sockfd, buf, count, 0),
        ))
        .unwrap()
    };
    assert_eq!(&buffer, TEST_BYTES);

    // Test non-blocking writing.

    // Sending into the empty buffer should succeed without blocking.
    unsafe {
        errno_result(libc_utils::write_all_generic(
            TEST_BYTES.as_ptr().cast(),
            TEST_BYTES.len(),
            libc_utils::NoRetry,
            |buf, count| libc::send(client_sockfd, buf, count, 0),
        ))
        .unwrap()
    };

    if !cfg!(windows_host) {
        // Keep sending data until the buffer is full and we block.
        // We cannot test this on Windows since there apparently the send buffer
        // never fills up, at least for localhost connections.

        let fill_buf = [1u8; 5_000_000];
        // This fills the socket receive buffer and thus should start blocking.
        let err = unsafe {
            errno_result(libc_utils::write_all_generic(
                fill_buf.as_ptr().cast(),
                fill_buf.len(),
                libc_utils::NoRetry,
                |buf, count| libc::send(client_sockfd, buf, count, 0),
            ))
            .unwrap_err()
        };
        assert_eq!(err.kind(), ErrorKind::WouldBlock)
    }

    server_thread.join().unwrap();
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "solaris",
    target_os = "illumos"
))]
/// Test sending bytes into and receiving bytes from a connected stream without blocking.
/// Instead of using non-blocking sockets, we test whether it works with blocking sockets
/// when passing the `libc::MSG_DONTWAIT` flag to the send and receive calls.
fn test_send_recv_dontwait() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();
        // Similar to above we use blocking operations on the server side.

        // Yield back to client so that it starts receiving before we start sending.
        thread::sleep(Duration::from_millis(10));

        unsafe {
            errno_result(libc_utils::write_all_generic(
                TEST_BYTES.as_ptr().cast(),
                TEST_BYTES.len(),
                libc_utils::NoRetry,
                |buf, count| libc::send(peerfd, buf, count, 0),
            ))
            .unwrap()
        };

        // The buffer should contain `TEST_BYTES` at the beginning.
        // This will block until the client sent us this data.
        let mut buffer = [0; TEST_BYTES.len()];
        unsafe {
            errno_result(libc_utils::read_all_generic(
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                libc_utils::NoRetry,
                |buf, count| libc::recv(peerfd, buf, count, 0),
            ))
            .unwrap()
        };
        assert_eq!(&buffer, TEST_BYTES);
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    // We are connected and the server socket is not writing.

    let mut buffer = [0; TEST_BYTES.len()];
    // Receiving from a socket when the peer is not writing is
    // not possible without blocking.
    let err = unsafe {
        errno_result(libc::recv(
            client_sockfd,
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc::MSG_DONTWAIT,
        ))
        .unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Try to receive bytes from the peer socket without blocking.
    // Since the peer socket might do partial writes, we might need to
    // sleep multiple times until we received everything.

    unsafe {
        errno_result(libc_utils::read_all_generic(
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc_utils::RetryAfter(Duration::from_millis(10)),
            |buf, count| libc::recv(client_sockfd, buf, count, libc::MSG_DONTWAIT),
        ))
        .unwrap()
    };
    assert_eq!(&buffer, TEST_BYTES);

    // Test non-blocking writing.

    // Sending into the empty buffer should succeed without blocking.
    unsafe {
        errno_result(libc_utils::write_all_generic(
            TEST_BYTES.as_ptr().cast(),
            TEST_BYTES.len(),
            libc_utils::NoRetry,
            |buf, count| libc::send(client_sockfd, buf, count, libc::MSG_DONTWAIT),
        ))
        .unwrap()
    };

    if !cfg!(windows_host) {
        // Keep sending data until the buffer is full and we block.
        // We cannot test this on Windows since there apparently the send buffer
        // never fills up, at least for localhost connections.

        let fill_buf = [1u8; 5_000_000];
        // This fills the socket receive buffer and thus should start blocking.
        let err = unsafe {
            errno_result(libc_utils::write_all_generic(
                fill_buf.as_ptr().cast(),
                fill_buf.len(),
                libc_utils::NoRetry,
                |buf, count| libc::send(client_sockfd, buf, count, libc::MSG_DONTWAIT),
            ))
            .unwrap_err()
        };
        assert_eq!(err.kind(), ErrorKind::WouldBlock)
    }

    server_thread.join().unwrap();
}

/// Test writing bytes into and reading bytes from a connected stream without blocking.
fn test_write_read_nonblock() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();
        // Similar to above we use blocking operations on the server side.

        // Yield back to client so that it starts receiving before we start sending.
        thread::sleep(Duration::from_millis(10));

        let bytes_written = unsafe {
            errno_result(libc_utils::write_all(
                peerfd,
                TEST_BYTES.as_ptr().cast(),
                TEST_BYTES.len(),
            ))
            .unwrap()
        };
        assert_eq!(bytes_written as usize, TEST_BYTES.len());

        // The buffer should contain `TEST_BYTES` at the beginning.
        // This will block until the client sent us this data.
        let mut buffer = [0; TEST_BYTES.len()];
        unsafe {
            errno_result(libc_utils::read_all(peerfd, buffer.as_mut_ptr().cast(), buffer.len()))
                .unwrap()
        };
        assert_eq!(&buffer, TEST_BYTES);
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    unsafe {
        // Change client socket to be non-blocking.
        errno_check(libc::fcntl(client_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // We are connected and the server socket is not writing.

    let mut buffer = [0; TEST_BYTES.len()];
    // Reading from a socket when the peer is not writing is
    // not possible without blocking.
    let err = unsafe {
        errno_result(libc::read(
            client_sockfd,
            buffer.as_mut_ptr() as *mut libc::c_void,
            buffer.len(),
        ))
        .unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Try to read bytes from the peer socket without blocking.
    // Since the peer socket might do partial writes, we might need to
    // sleep multiple times until we read everything.

    unsafe {
        errno_result(libc_utils::read_all_generic(
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc_utils::RetryAfter(Duration::from_millis(10)),
            |buf, count| libc::read(client_sockfd, buf, count),
        ))
        .unwrap()
    };
    assert_eq!(&buffer, TEST_BYTES);

    // Now we test non-blocking writing.

    // Writing into the empty buffer should succeed without blocking.
    let bytes_written = unsafe {
        errno_result(libc_utils::write_all(
            client_sockfd,
            TEST_BYTES.as_ptr().cast(),
            TEST_BYTES.len(),
        ))
        .unwrap()
    };
    assert_eq!(bytes_written as usize, TEST_BYTES.len());

    if !cfg!(windows_host) {
        // Keep sending data until the buffer is full and we block.
        // We cannot test this on Windows since there apparently the send buffer
        // never fills up, at least for localhost connections.

        let fill_buf = [1u8; 5_000_000];
        // This fills the socket receive buffer and thus should start blocking.
        let err = unsafe {
            errno_result(libc_utils::write_all_generic(
                fill_buf.as_ptr().cast(),
                fill_buf.len(),
                libc_utils::NoRetry,
                |buf, count| libc::write(client_sockfd, buf, count),
            ))
            .unwrap_err()
        };
        assert_eq!(err.kind(), ErrorKind::WouldBlock)
    }

    server_thread.join().unwrap();
}

/// Test that the `getpeername` syscall successfully returns the peer address
/// for a non-blocking IPv4 socket whose connection has been successfully
/// established before calling the syscall.
fn test_getpeername_ipv4_nonblock() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    unsafe {
        // Change client socket to be non-blocking.
        errno_check(libc::fcntl(client_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        net::accept_ipv4(server_sockfd).unwrap();
    });

    // Yield to server thread to ensure that it's currently accepting.
    thread::sleep(Duration::from_millis(10));

    // Non-blocking connects always "fail" with EINPROGRESS.
    let err = net::connect_ipv4(client_sockfd, addr).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InProgress);

    loop {
        let peername_result = net::sockname_ipv4(|storage, len| unsafe {
            libc::getpeername(client_sockfd, storage, len)
        });

        match peername_result {
            Ok((_, peer_addr)) => {
                assert_eq!(addr.sin_family, peer_addr.sin_family);
                assert_eq!(addr.sin_port, peer_addr.sin_port);
                assert_eq!(addr.sin_addr.s_addr, peer_addr.sin_addr.s_addr);
                break;
            }
            Err(err) if err.kind() == ErrorKind::NotConnected => {
                // Connection is not yet established; wait and retry later.
                thread::sleep(Duration::from_millis(10))
            }
            Err(err) => {
                panic!("error whilst getting peername: {err}")
            }
        }
    }

    server_thread.join().unwrap();
}

/// Test that the `getpeername` syscall returns ENOTCONN
/// for a non-blocking IPv4 socket which is stuck at
/// connecting to the remote address.
fn test_getpeername_ipv4_nonblock_no_peer() {
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    unsafe {
        // Change client socket to be non-blocking.
        errno_check(libc::fcntl(client_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // We cannot attempt to connect to a localhost address because
    // it could be the case that a socket from another test is
    // currently listening on `localhost:12321` because we bind to
    // random ports everywhere. For `192.0.2.1` we know that nothing is
    // listening because it's a blackhole address:
    // <https://www.rfc-editor.org/rfc/rfc5737>
    // The port `12321` is just a random non-zero port because Windows
    // and Apple hosts return EADDRNOTAVAIL when attempting to connect to
    // a zero port.
    let addr = net::sock_addr_ipv4([192, 0, 2, 1], 12321);

    // Non-blocking connect should fail with EINPROGRESS.
    let err = net::connect_ipv4(client_sockfd, addr).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InProgress);

    // Since we're never accepting the connection, the socket should never be
    // successfully connected and thus we should be unable to read the peername.
    let Err(err) = net::sockname_ipv4(|storage, len| unsafe {
        libc::getpeername(client_sockfd, storage, len)
    }) else {
        unreachable!()
    };
    assert_eq!(err.kind(), ErrorKind::NotConnected);
}
