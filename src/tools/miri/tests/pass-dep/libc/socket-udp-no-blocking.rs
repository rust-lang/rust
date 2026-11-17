//@ignore-target: windows  # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation
//@run-native

//@revisions: bsd_host other_host
//@[bsd_host] only-host: macos freebsd
//@[other_host] ignore-host: macos freebsd

#[path = "../../utils/libc.rs"]
mod libc_utils;

use std::io::ErrorKind;
use std::time::Duration;
use std::{ptr, thread};

use libc_utils::*;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_send_peek_recv_nonblock();
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "solaris",
        target_os = "illumos"
    ))]
    test_send_recv_dontwait();
    test_write_read_nonblock();
    test_send_not_connected();
    test_send_message_too_big();
    test_recv_discard_excess();
    test_sendto_recvfrom();
    test_sendto_implicit_bind();
}

/// Test sending bytes into an UDP socket and peeking and receiving them from another UDP socket
/// without blocking.
fn test_send_peek_recv_nonblock() {
    let (rx_sockfd, addr) = net::make_local_ipv4(libc::SOCK_DGRAM).unwrap();
    let tx_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };

    unsafe {
        // Change both sockets to be non-blocking.
        errno_check(libc::fcntl(rx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
        errno_check(libc::fcntl(tx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Set the default destination address of the transmitting socket.
    net::connect_ipv4(tx_sockfd, addr).unwrap();

    let mut buffer = [0; TEST_BYTES.len()];

    // Receiving from a socket when no data is available is not possible without blocking.
    let err = unsafe {
        errno_result(libc::recv(rx_sockfd, buffer.as_mut_ptr().cast(), buffer.len(), 0))
            .unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Write `TEST_BYTES` into the transmitting socket.
    let bytes_sent = unsafe {
        errno_result(libc::send(tx_sockfd, TEST_BYTES.as_ptr().cast(), TEST_BYTES.len(), 0))
            .unwrap()
    };
    assert_eq!(bytes_sent as usize, TEST_BYTES.len());

    // Peek `TEST_BYTES` from the receiving socket.
    // On BSDs, UDP sockets appear to be asynchronous, so we have a loop that waits for the packets
    // that we just sent above.
    loop {
        let result = unsafe {
            errno_result(libc::recv(
                rx_sockfd,
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                libc::MSG_PEEK,
            ))
        };
        match result {
            Ok(bytes_received) => {
                assert_eq!(bytes_received as usize, buffer.len());
                break;
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(10)),
            Err(e) => panic!("error while receiving: {e}"),
        }
    }
    assert_eq!(&buffer, TEST_BYTES);

    // The peek shouldn't have removed the data from the socket; we should
    // thus be able to read it again.

    let mut buffer = [0; TEST_BYTES.len()];

    // Read `TEST_BYTES` from the receiving socket.
    let bytes_received = unsafe {
        errno_result(libc::recv(rx_sockfd, buffer.as_mut_ptr().cast(), buffer.len(), 0)).unwrap()
    };
    assert_eq!(bytes_received as usize, TEST_BYTES.len());
    assert_eq!(&buffer, TEST_BYTES);
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "solaris",
    target_os = "illumos"
))]
/// Test sending bytes into an UDP socket and receiving them from another UDP socket without blocking.
/// In this case, the sockets are blocking sockets, however, all `recv` and `send` operations have
/// the `libc::MSG_DONTWAIT` flag set which means that those operations should be performed in a
/// non-blocking manner.
fn test_send_recv_dontwait() {
    let (rx_sockfd, addr) = net::make_local_ipv4(libc::SOCK_DGRAM).unwrap();
    let tx_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };

    // Set the default destination address of the transmitting socket.
    net::connect_ipv4(tx_sockfd, addr).unwrap();

    let mut buffer = [0; TEST_BYTES.len()];

    // Write `TEST_BYTES` into the transmitting socket.
    let bytes_sent = unsafe {
        errno_result(libc::send(
            tx_sockfd,
            TEST_BYTES.as_ptr().cast(),
            TEST_BYTES.len(),
            libc::MSG_DONTWAIT,
        ))
        .unwrap()
    };
    assert_eq!(bytes_sent as usize, TEST_BYTES.len());

    // Read `TEST_BYTES` from the receiving socket.
    // On BSDs, UDP sockets appear to be asynchronous, so we have a loop that waits for the packets
    // that we just sent above.
    loop {
        let result = unsafe {
            errno_result(libc::recv(
                rx_sockfd,
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                libc::MSG_DONTWAIT,
            ))
        };
        match result {
            Ok(bytes_received) => {
                assert_eq!(bytes_received as usize, TEST_BYTES.len());
                break;
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(10)),
            Err(e) => panic!("error while reading: {e}"),
        }
    }
    assert_eq!(&buffer, TEST_BYTES);
}

/// Test writing bytes into an UDP socket and reading them from another UDP socket without
/// blocking.
fn test_write_read_nonblock() {
    let (rx_sockfd, addr) = net::make_local_ipv4(libc::SOCK_DGRAM).unwrap();
    let tx_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };

    unsafe {
        // Change both sockets to be non-blocking.
        errno_check(libc::fcntl(rx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
        errno_check(libc::fcntl(tx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Set the default destination address of the transmitting socket.
    net::connect_ipv4(tx_sockfd, addr).unwrap();

    let mut buffer = [0; TEST_BYTES.len()];

    // Reading from a socket when no data is available is not possible without blocking.
    let err = unsafe {
        errno_result(libc::read(rx_sockfd, buffer.as_mut_ptr().cast(), buffer.len())).unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    // Write `TEST_BYTES` into the transmitting socket.
    let bytes_written = unsafe {
        errno_result(libc::write(tx_sockfd, TEST_BYTES.as_ptr().cast(), TEST_BYTES.len())).unwrap()
    };
    assert_eq!(bytes_written as usize, TEST_BYTES.len());

    // Read `TEST_BYTES` from the receiving socket.
    // On BSDs, UDP sockets appear to be asynchronous, so we have a loop that waits for the packets
    // that we just sent above.
    loop {
        let result = unsafe {
            errno_result(libc::read(rx_sockfd, buffer.as_mut_ptr().cast(), buffer.len()))
        };
        match result {
            Ok(bytes_read) => {
                assert_eq!(bytes_read as usize, TEST_BYTES.len());
                break;
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(10)),
            Err(e) => panic!("error while reading: {e}"),
        }
    }
    assert_eq!(&buffer, TEST_BYTES);
}

/// Test sending bytes into a UDP socket with `send` but there hasn't been set a
/// default destination using the `connect` function. This should fail with EDESTADDRREQ.
fn test_send_not_connected() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };

    unsafe {
        // Change the socket to be non-blocking.
        errno_check(libc::fcntl(sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Attempt to write `TEST_BYTES` into the socket.
    let err = unsafe {
        errno_result(libc::send(sockfd, TEST_BYTES.as_ptr().cast(), TEST_BYTES.len(), 0))
            .unwrap_err()
    };
    if cfg!(miri) {
        // On Windows hosts the `send` fails with ENOTCONN while on Unix-like hosts
        // it fails with EDESTADDRREQ since the socket has no default destination address
        // set. We don't explicitly remap the errors and thus accept both here.
        assert!(matches!(err.raw_os_error().unwrap(), libc::EDESTADDRREQ | libc::ENOTCONN));
    } else {
        // On Unix-like systems the `send` should fail with EDESTADDRREQ since
        // the socket has no default destination address set.
        assert_eq!(err.raw_os_error().unwrap(), libc::EDESTADDRREQ);
    }
}

/// Test sending a UDP message which doesn't fit into a single UDP datagram.
/// The `send` operation should fail with EMSGSIZE.
fn test_send_message_too_big() {
    let (_rx_sockfd, addr) = net::make_local_ipv4(libc::SOCK_DGRAM).unwrap();
    let tx_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };

    unsafe {
        // Change the transmitting socket to be non-blocking.
        errno_check(libc::fcntl(tx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Set the default destination address of the transmitting socket.
    net::connect_ipv4(tx_sockfd, addr).unwrap();

    // The maximum UDP packet size is `u16::MAX` which is less than 70'000 bytes.
    let buffer = [1u8; 70_000];

    let err = unsafe {
        errno_result(libc::send(tx_sockfd, buffer.as_ptr().cast(), buffer.len(), 0)).unwrap_err()
    };
    // `send` should fail with EMSGSIZE since the message doesn't fit into a single UDP datagram.
    assert_eq!(err.raw_os_error(), Some(libc::EMSGSIZE));
}

/// Test that reading a partial message discards the remaining bytes of the message
/// i.e. they cannot be read by a subsequent `recv`.
fn test_recv_discard_excess() {
    let (rx_sockfd, addr) = net::make_local_ipv4(libc::SOCK_DGRAM).unwrap();
    let tx_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };

    unsafe {
        // Change both sockets to be non-blocking.
        errno_check(libc::fcntl(rx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
        errno_check(libc::fcntl(tx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Set the default destination address of the transmitting socket.
    net::connect_ipv4(tx_sockfd, addr).unwrap();

    let mut buffer = [0u8; TEST_BYTES.len() / 2];

    // Write `TEST_BYTES` into the transmitting socket.
    unsafe {
        errno_result(libc::write(tx_sockfd, TEST_BYTES.as_ptr().cast(), TEST_BYTES.len())).unwrap();
    }

    // Read `TEST_BYTES` partially from the receiving socket.
    // On BSDs, UDP sockets appear to be asynchronous, so we have a loop that waits for the packets
    // that we just sent above.
    loop {
        let result = unsafe {
            errno_result(libc::recv(rx_sockfd, buffer.as_mut_ptr().cast(), buffer.len(), 0))
        };
        match result {
            Ok(bytes_received) => {
                assert_eq!(bytes_received as usize, buffer.len());
                break;
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(10)),
            Err(e) => panic!("error while receiving: {e}"),
        }
    }
    assert_eq!(&buffer, &TEST_BYTES[0..buffer.len()]);

    // The bytes from the message which didn't fit into the buffer are discarded.
    // The read buffer should thus now be empty.
    let err = unsafe {
        errno_result(libc::recv(rx_sockfd, buffer.as_mut_ptr().cast(), buffer.len(), 0))
            .unwrap_err()
    };
    assert_eq!(err.kind(), ErrorKind::WouldBlock);
}

/// Test `sendto` and `recvfrom` with various combinations for the destination/origin address
/// pointer.
/// - `sendto` where the destination address is a nullptr and without a default destination address:
///   Should fail with EDESTADDRREQ.
/// - `sendto` where the destination address is valid and the default destination address is unavailable:
///   Should succeed on non-BSD targets since the `sendto` address should overwrite the default destination address.
///   Should fail with EISCONN on BSD targets.
/// - `sendto` where the destination address is a nullptr and with a default destination address:
///   Should succeed.
/// - `sendto` where the destination address is valid address:
///   Should succeed.
/// - `recvfrom` where the origin address pointer is a nullptr:
///   Should succeed.
/// - `recvfrom` where the origin address pointer is a valid pointer:
///   Should succeed.
fn test_sendto_recvfrom() {
    let (rx_sockfd, rx_addr) = net::make_local_ipv4(libc::SOCK_DGRAM).unwrap();
    let (tx_sockfd, tx_addr) = net::make_local_ipv4(libc::SOCK_DGRAM).unwrap();
    let other_tx_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };

    unsafe {
        // Change all sockets to be non-blocking.
        errno_check(libc::fcntl(rx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
        errno_check(libc::fcntl(tx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
        errno_check(libc::fcntl(other_tx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Write `TEST_BYTES` without specifying an explicit destination address
    // and without having a default destination address configured.
    // This should fail with EDESTADDRREQ.
    let err = unsafe {
        errno_result(libc::sendto(
            tx_sockfd,
            TEST_BYTES.as_ptr().cast(),
            TEST_BYTES.len(),
            /* flags */ 0,
            ptr::null::<libc::sockaddr_in>().cast(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ))
        .unwrap_err()
    };
    assert_eq!(err.raw_os_error().unwrap(), libc::EDESTADDRREQ);

    // Send `TEST_BYTES` from the transmitting socket to the receiving socket
    // by specifying `rx_addr` as the destination address of `sendto`.
    let bytes_sent = unsafe {
        errno_result(libc::sendto(
            tx_sockfd,
            TEST_BYTES.as_ptr().cast(),
            TEST_BYTES.len(),
            /* flags */ 0,
            (&rx_addr as *const libc::sockaddr_in).cast(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ))
        .unwrap()
    };
    assert_eq!(bytes_sent as usize, TEST_BYTES.len());

    // Set default destination address to `rx_addr`.
    net::connect_ipv4(tx_sockfd, rx_addr).unwrap();

    // When a default destination address is specified using `connect`, then
    // a `sendto` with a nullptr destination address falls back to the default
    // destination address.
    let bytes_sent = unsafe {
        errno_result(libc::sendto(
            tx_sockfd,
            TEST_BYTES.as_ptr().cast(),
            TEST_BYTES.len(),
            /* flags */ 0,
            ptr::null::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ))
        .unwrap()
    };
    assert_eq!(bytes_sent as usize, TEST_BYTES.len());

    // Set default destination address to some bogus address that is not available.
    // Since we cannot use `connect` multiple times on `tx_sockfd` we use `other_tx_sockfd`
    // for this.
    let bogus_addr = net::sock_addr_ipv4([127, 0, 3, 1], 5124);
    net::connect_ipv4(other_tx_sockfd, bogus_addr).unwrap();

    if !cfg!(any(bsd_host, target_os = "freebsd", target_vendor = "apple")) {
        // We're on a non-BSD host and on a non-BSD target.

        // A `sendto` with the destination address of `rx_addr` should overwrite the
        // bogus default destination address and thus `TEST_BYTES` should be transmitted
        // successfully from the transmitting socket to the receiving socket.
        let bytes_sent = unsafe {
            errno_result(libc::sendto(
                other_tx_sockfd,
                TEST_BYTES.as_ptr().cast(),
                TEST_BYTES.len(),
                /* flags */ 0,
                (&rx_addr as *const libc::sockaddr_in).cast(),
                size_of::<libc::sockaddr_in>() as libc::socklen_t,
            ))
            .unwrap()
        };
        assert_eq!(bytes_sent as usize, TEST_BYTES.len());
    } else if cfg!(any(target_os = "freebsd", target_vendor = "apple")) {
        // We're on a BSD target.

        // A `sendto` with a non-null destination address and a configured default
        // destination address should return EISCONN.
        let err = unsafe {
            errno_result(libc::sendto(
                other_tx_sockfd,
                TEST_BYTES.as_ptr().cast(),
                TEST_BYTES.len(),
                /* flags */ 0,
                (&rx_addr as *const libc::sockaddr_in).cast(),
                size_of::<libc::sockaddr_in>() as libc::socklen_t,
            ))
            .unwrap_err()
        };
        assert_eq!(err.raw_os_error().unwrap(), libc::EISCONN);
    }

    let mut buffer = [0u8; TEST_BYTES.len()];

    // Read `TEST_BYTES` from the receiving socket using `recvfrom` and ensure that the origin
    // address matches `tx_addr`.
    // On BSDs, UDP sockets appear to be asynchronous, so we have a loop that waits for the packets
    // that we just sent above.
    loop {
        let result = net::sockname_ipv4(|storage, len| unsafe {
            libc::recvfrom(
                rx_sockfd,
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                /* flags */ 0,
                storage,
                len,
            )
        });
        match result {
            Ok((bytes_received, origin_addr)) => {
                assert_eq!(bytes_received as usize, TEST_BYTES.len());
                assert_eq!(origin_addr.sin_family, tx_addr.sin_family);
                assert_eq!(origin_addr.sin_port, tx_addr.sin_port);
                assert_eq!(origin_addr.sin_addr.s_addr, tx_addr.sin_addr.s_addr);
                break;
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(10)),
            Err(e) => panic!("error while reading: {e}"),
        }
    }
    assert_eq!(&buffer, TEST_BYTES);

    let mut buffer = [0u8; TEST_BYTES.len()];

    // Read `TEST_BYTES` from the receiving socket using `recvfrom` but with a nullptr address pointer.
    // This indicates that we don't care about the origin address and thus it should still succeed.
    // On BSDs, UDP sockets appear to be asynchronous, so we have a loop that waits for the packets
    // that we just sent above.
    loop {
        let result = unsafe {
            errno_result(libc::recvfrom(
                rx_sockfd,
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                /* flags */ 0,
                ptr::null_mut::<libc::sockaddr>(),
                ptr::null_mut::<libc::socklen_t>(),
            ))
        };
        match result {
            Ok(bytes_received) => {
                assert_eq!(bytes_received as usize, TEST_BYTES.len());
                break;
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(10)),
            Err(e) => panic!("error while reading: {e}"),
        }
    }
    assert_eq!(&buffer, TEST_BYTES);
}

/// Test that a UDP socket gets implicitly bound to a local address after
/// the first `sendto` has been invoked on it.
fn test_sendto_implicit_bind() {
    let (rx_sockfd, rx_addr) = net::make_local_ipv4(libc::SOCK_DGRAM).unwrap();
    let tx_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };

    unsafe {
        // Change both sockets to be non-blocking.
        errno_check(libc::fcntl(rx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
        errno_check(libc::fcntl(tx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    let bytes_sent = unsafe {
        errno_result(libc::sendto(
            tx_sockfd,
            TEST_BYTES.as_ptr().cast(),
            TEST_BYTES.len(),
            /* flags */ 0,
            (&rx_addr as *const libc::sockaddr_in).cast(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ))
        .unwrap()
    };
    assert_eq!(bytes_sent as usize, TEST_BYTES.len());

    // After the first `sendto` the `tx_sockfd` should be implicitly bound to a local address.
    // This means `sock_addr.sin_port` should be non-zero. A zero port would indicate that
    // the socket hasn't been bound.
    let (_, sock_addr) =
        net::sockname_ipv4(|storage, len| unsafe { libc::getsockname(tx_sockfd, storage, len) })
            .unwrap();

    // Libc representation of an unspecified IPv4 address with zero port.
    let addr = net::sock_addr_ipv4([0, 0, 0, 0], 0);

    // Ensure that the socket got implicitly bound to the unspecified
    // local address with a non-zero port.
    assert!(sock_addr.sin_port > 0);
    assert_eq!(addr.sin_family, sock_addr.sin_family);
    assert_eq!(addr.sin_addr.s_addr, sock_addr.sin_addr.s_addr);
}
