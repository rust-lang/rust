//@only-target: linux android illumos
//@compile-flags: -Zmiri-disable-isolation

#![feature(io_error_inprogress)]

#[path = "../../utils/libc.rs"]
mod libc_utils;

use std::io::ErrorKind;
use std::thread;
use std::time::Duration;

use libc_utils::epoll::*;
use libc_utils::*;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_connect_nonblock();
    test_accept_nonblock();
    test_connect_nonblock_err();
    test_recv_nonblock();
    test_send_nonblock();
    test_shutdown_read_write();
    test_shutdown_read();
    test_shutdown_write();
    test_readiness_after_short_read();
    test_readiness_after_short_write();
}

/// Test that connecting to a server socket works when the client
/// socket is non-blocking before the `connect` call.
/// Instead of busy waiting until we no longer get ENOTCONN, we register
/// the client socket to epoll and wait for a WRITABLE event.
fn test_connect_nonblock() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    unsafe {
        // Change client socket to be non-blocking.
        errno_check(libc::fcntl(client_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        // Yield to client thread to ensure that it actually needs
        // to wait until the connection gets accepted.
        thread::sleep(Duration::from_millis(10));

        net::accept_ipv4(server_sockfd).unwrap();
    });

    // Non-blocking connects always "fail" with EINPROGRESS.
    let err = net::connect_ipv4(client_sockfd, addr).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InProgress);

    // Add client socket with WRITABLE interest to epoll.
    epoll_ctl_add(epfd, client_sockfd, EPOLLOUT | EPOLLET | EPOLLERR).unwrap();

    // Wait until we are done connecting.
    check_epoll_wait::<8>(epfd, &[Ev { events: EPOLLOUT, data: client_sockfd }], -1);

    // There should be no error during async connection.
    let errno =
        net::getsockopt::<libc::c_int>(client_sockfd, libc::SOL_SOCKET, libc::SO_ERROR).unwrap();
    assert_eq!(errno, 0);

    // We should now be connected and thus getting the peer name should work.
    net::sockname_ipv4(|storage, len| unsafe { libc::getpeername(client_sockfd, storage, len) })
        .unwrap();

    server_thread.join().unwrap();
}

/// Test that accepting incoming connections works with non-blocking server sockets.
/// Instead of busy waiting until we no longer get EWOULDBLOCK, we add the
/// server socket to an epoll instance and wait for a READABLE event.
fn test_accept_nonblock() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    unsafe {
        // Change server socket to be non-blocking.
        errno_check(libc::fcntl(server_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        // Add client socket with READABLE interest to epoll.
        epoll_ctl_add(epfd, server_sockfd, EPOLLIN | EPOLLET | EPOLLERR).unwrap();

        // Wait until we get a readable event on the server socket.
        check_epoll_wait::<8>(epfd, &[Ev { events: EPOLLIN, data: server_sockfd }], -1);

        // Accepting should now be possible.
        net::accept_ipv4(server_sockfd).unwrap();

        // Accepting should now trigger an EWOULDBLOCK.
        let err = net::accept_ipv4(server_sockfd).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::WouldBlock);

        // Ensure that the server is no longer readable because we just
        // attempted to accept without an incoming connection.
        assert_eq!(current_epoll_readiness::<8>(server_sockfd, EPOLLIN | EPOLLET), 0);
    });

    // Yield to server thread such that it actually needs to wait for an
    // incoming client connection.
    thread::sleep(Duration::from_millis(10));

    net::connect_ipv4(client_sockfd, addr).unwrap();

    server_thread.join().unwrap();
}

/// Test that the SO_ERROR socket option is set when attempting to
/// connect to an unbound address without blocking.
fn test_connect_nonblock_err() {
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
    check_epoll_wait::<8>(
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
    let readiness = current_epoll_readiness::<8>(client_sockfd, EPOLLET | EPOLLOUT | EPOLLERR);
    assert!(readiness & EPOLLERR == 0);
}

/// Test receiving bytes from a connected stream without blocking.
/// Instead of busy waiting until we no longer receive EWOULDBLOCK when trying to
/// read from the client, we register the client socket to epoll and wait for
/// READABLE events.
fn test_recv_nonblock() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();
        // `peerfd` is a blocking socket now. But that's okay, the client still does non-blocking
        // reads/writes.

        // Yield back to client so that it starts receiving before we start sending.
        thread::sleep(Duration::from_millis(10));

        libc_utils::write_all(peerfd, TEST_BYTES).unwrap();
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
    // call `epoll_wait` multiple times until we received everything.

    // Add client socket with READABLE interest to epoll.
    epoll_ctl_add(epfd, client_sockfd, EPOLLIN | EPOLLET | EPOLLERR).unwrap();

    let mut bytes_received = 0;

    while bytes_received != buffer.len() {
        let read_result = unsafe {
            errno_result(libc::recv(
                client_sockfd,
                buffer.as_mut_ptr().byte_add(bytes_received).cast(),
                buffer.len() - bytes_received,
                0,
            ))
        };

        match read_result {
            Ok(received) => bytes_received += received as usize,
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                // Use epoll to block until there's data available again.
                check_epoll_wait::<8>(epfd, &[Ev { events: EPOLLIN, data: client_sockfd }], -1);
            }
            Err(err) => panic!("unexpected error whilst receiving: {err}"),
        }
    }

    assert_eq!(&buffer, TEST_BYTES);

    let mut buffer = [0u8; 1];
    let err = unsafe {
        errno_result(libc::recv(client_sockfd, buffer.as_mut_ptr().cast(), buffer.len(), 0))
            .unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Ensure that the client is no longer readable because
    // we just got an EWOULDBLOCK from a receive call.
    assert_eq!(current_epoll_readiness::<8>(client_sockfd, EPOLLIN | EPOLLET), 0);

    server_thread.join().unwrap();
}

/// Test sending bytes into a connected stream without blocking.
/// Once the buffer is filled we wait for the epoll WRITABLE
/// readiness instead of busy waiting until we can
/// write again.
fn test_send_nonblock() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();
        peerfd
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    unsafe {
        // Change client socket to be non-blocking.
        errno_check(libc::fcntl(client_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Try to send bytes to the peer socket without blocking.
    // We first fill up the buffer and ensure that we keep the
    // writable readiness during the fill up process.

    // Add client socket with READABLE interest to epoll.
    epoll_ctl_add(epfd, client_sockfd, EPOLLOUT | EPOLLET | EPOLLERR).unwrap();

    let fill_buf = [1u8; 32_000];
    let mut total_written = 0usize;
    loop {
        let result = unsafe {
            errno_result(libc::send(client_sockfd, fill_buf.as_ptr().cast(), fill_buf.len(), 0))
        };

        match result {
            Ok(written) => {
                total_written += written as usize;
                if written as usize == fill_buf.len() {
                    // When we didn't have a short write we should still be able to write more.
                    // Ensure the socket is still writable.
                    assert_eq!(
                        current_epoll_readiness::<8>(client_sockfd, EPOLLOUT | EPOLLET),
                        EPOLLOUT
                    );
                }
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => break,
            Err(err) => panic!("unexpected error whilst filling up buffer: {err}"),
        }
    }

    // The buffer should be filled up because we received an EWOULDBLOCK.
    // This also means that the client socket should no longer be writable.
    assert_eq!(current_epoll_readiness::<8>(client_sockfd, EPOLLOUT | EPOLLET), 0);

    let peerfd = server_thread.join().unwrap();

    // Spawn the reader thread.
    let reader_thread = thread::spawn(move || {
        // Read half of the bytes needed to fill the buffer.
        // This should make the client buffer writable again.
        // We need to do it this way because different hosts
        // have different read thresholds after which they
        // trigger a new writable event.
        let mut buffer = Vec::with_capacity(total_written / 2);
        buffer.fill(0u8);
        unsafe {
            libc_utils::read_exact_generic(
                buffer.as_mut_ptr().cast(),
                total_written / 2,
                libc_utils::NoRetry,
                |buf, count| libc::recv(peerfd, buf, count, 0),
            )
            .unwrap()
        };
    });

    // Wait until the socket is again writable.
    check_epoll_wait::<8>(epfd, &[Ev { events: EPOLLOUT, data: client_sockfd }], -1);

    let fill_buf = [1u8; 100];
    // We should be able to write again without blocking because we just received
    // a writable event.
    unsafe {
        errno_result(libc::send(client_sockfd, fill_buf.as_ptr().cast(), fill_buf.len(), 0))
            .unwrap()
    };

    reader_thread.join().unwrap();
}

/// Test that the EPOLLHUP and EPOLLRDHUP readiness are set when both
/// the read and write ends of a socket are closed.
fn test_shutdown_read_write() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let epfd = unsafe { libc::epoll_create1(0) };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || net::accept_ipv4(server_sockfd).unwrap());

    net::connect_ipv4(client_sockfd, addr).unwrap();

    epoll_ctl_add(epfd, client_sockfd, EPOLLET | EPOLLHUP | EPOLLRDHUP | EPOLLIN).unwrap();

    // Close the read and write end of the socket.
    unsafe { libc::shutdown(client_sockfd, libc::SHUT_RDWR) };

    // Ensure that the "read end closed", "write end closed", and "readable" readiness are set.
    check_epoll_wait::<8>(
        epfd,
        &[Ev { events: EPOLLRDHUP | EPOLLHUP | EPOLLIN, data: client_sockfd }],
        -1,
    );

    server_thread.join().unwrap();
}

/// Test that the EPOLLRDHUP readiness is set when the read
/// end of a socket is closed.
fn test_shutdown_read() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let epfd = unsafe { libc::epoll_create1(0) };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || net::accept_ipv4(server_sockfd).unwrap());

    net::connect_ipv4(client_sockfd, addr).unwrap();

    epoll_ctl_add(epfd, client_sockfd, EPOLLET | EPOLLHUP | EPOLLRDHUP).unwrap();

    // Close the read end of the socket.
    unsafe { libc::shutdown(client_sockfd, libc::SHUT_RD) };

    // Ensure that the "read end closed" readiness is set.
    check_epoll_wait::<8>(epfd, &[Ev { events: EPOLLRDHUP, data: client_sockfd }], -1);

    server_thread.join().unwrap();
}

/// Test that the EPOLLRDHUP readiness is set when the write
/// end of the peer socket is closed.
fn test_shutdown_write() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let epfd = unsafe { libc::epoll_create1(0) };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();
        // Close the write end of the peer socket.
        unsafe { libc::shutdown(peerfd, libc::SHUT_WR) };
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    epoll_ctl_add(epfd, client_sockfd, EPOLLET | EPOLLHUP | EPOLLRDHUP).unwrap();

    // Ensure that the "read end closed" readiness is set when
    // the write end of the peer is closed.
    check_epoll_wait::<8>(epfd, &[Ev { events: EPOLLRDHUP, data: client_sockfd }], -1);

    server_thread.join().unwrap();
}

/// Test that Miri correctly removes the readable readiness or emits a new edge after a short read.
fn test_readiness_after_short_read() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();

        // Write `TEST_BYTES` into the stream.
        libc_utils::write_all(peerfd, TEST_BYTES).unwrap();

        // Return the peer socket file descriptor such that we can use
        // it after joining the server thread.
        peerfd
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    unsafe {
        // Change client socket to be non-blocking.
        errno_check(libc::fcntl(client_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Add client socket with readable interest to epoll.
    epoll_ctl_add(epfd, client_sockfd, EPOLLET | EPOLLIN).unwrap();

    // Wait until the socket becomes readable.
    check_epoll_wait::<8>(epfd, &[Ev { events: EPOLLIN, data: client_sockfd }], -1);

    let mut buffer = [0u8; 1024];

    // We want to read in chunks of 16 bytes. To ensure we get a short read, `TEST_BYTES.len()`
    // must not be dividable by 16.
    assert!(TEST_BYTES.len() % 16 != 0);

    let mut total_bytes_read = 0;
    // Read everything from the socket until we get a short read.
    // We don't want to provide `TEST_BYTES.len()` as `count` because then we won't trigger
    // a short read.
    loop {
        let bytes_read = unsafe {
            errno_result(libc::read(
                client_sockfd,
                buffer.as_mut_ptr().byte_add(total_bytes_read).cast(),
                // Read a chunk of 16 bytes.
                16,
            ))
            .unwrap()
        };

        total_bytes_read += bytes_read as usize;
        if bytes_read < 16 {
            // We had a short read; we thus assume the read buffer is empty.
            break;
        }
    }
    assert_eq!(total_bytes_read, TEST_BYTES.len());

    // We had a short read because `buffer` is bigger than `TEST_BYTES`.

    // The read buffer is now empty. We thus write again `TEST_BYTES` into the peer socket.
    let peerfd = server_thread.join().unwrap();
    libc_utils::write_all(peerfd, TEST_BYTES).unwrap();

    // Wait until the client socket becomes readable again.
    // If this blocks indefinitely, Miri lost track of the proper status of this socket.
    check_epoll_wait::<8>(epfd, &[Ev { events: EPOLLIN, data: client_sockfd }], -1);

    // Now we can read the 2nd chunk of data.
    unsafe {
        libc_utils::read_exact_generic(
            buffer.as_mut_ptr().cast(),
            TEST_BYTES.len(),
            libc_utils::NoRetry,
            |buf, count| libc::read(client_sockfd, buf, count),
        )
        .unwrap()
    };
}

/// Test that Miri correctly removes the writable readiness or emits a new edge after a short write.
fn test_readiness_after_short_write() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();
        // Return the peer socket file descriptor such that we can use
        // it after joining the server thread.
        peerfd
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    unsafe {
        // Change client socket to be non-blocking.
        errno_check(libc::fcntl(client_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // The peer socket is a blocking socket.
    let peerfd = server_thread.join().unwrap();

    // Add client socket with writable interest to epoll.
    epoll_ctl_add(epfd, client_sockfd, EPOLLET | EPOLLOUT).unwrap();

    // Wait until the socket becomes writable.
    check_epoll_wait::<8>(epfd, &[Ev { events: EPOLLOUT, data: client_sockfd }], -1);

    // We now want to fill the write buffer of the socket by repeatedly writing
    // `buffer` into it. The last write should then be a short write.
    // We assume/hope that the write buffer length is not divisible by 1039.
    let buffer = [123u8; 1039];

    let mut total_bytes_written = 0;
    loop {
        let result = unsafe {
            errno_result(libc::write(client_sockfd, buffer.as_ptr().cast(), buffer.len()))
        };

        match result {
            Ok(bytes_written) => {
                total_bytes_written += bytes_written as usize;
                if (bytes_written as usize) < buffer.len() {
                    // We had a short write; we thus assume the write buffer is full.
                    break;
                }
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                // Windows and Apple hosts behave weirdly when attempting to fill up the write buffer.
                // Instead of doing a short write to completely fill the buffer, they can return an
                // EWOULDBLOCK when the next write wouldn't fit into the buffer.
                // When we get such an error, we also assume the write buffer is full.
                break;
            }
            Err(err) => panic!("unexpected error whilst filling up buffer: {err}"),
        }
    }

    // By reading half of the written bytes on the peer socket, the client socket
    // should become writable again.
    let mut read_buffer = Vec::<u8>::with_capacity(total_bytes_written / 2);
    unsafe {
        libc_utils::read_exact_generic(
            read_buffer.as_mut_ptr().cast(),
            total_bytes_written / 2,
            libc_utils::NoRetry,
            |buf, count| libc::read(peerfd, buf, count),
        )
        .unwrap()
    };

    // Wait until the socket becomes writable again.
    // If this blocks indefinitely, Miri lost track of the proper status of this socket.
    check_epoll_wait::<8>(epfd, &[Ev { events: EPOLLOUT, data: client_sockfd }], -1);

    // We should again be able to write into the socket.
    libc_utils::write_all(client_sockfd, &buffer).unwrap();
}
