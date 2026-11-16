//@ignore-target: windows # No socket support on Windows
//@compile-flags: -Zmiri-disable-isolation
//@run-native

use std::net::UdpSocket;
use std::time::Duration;

fn main() {
    test_create_ipv4();
    test_create_ipv6();
    test_connect();
    test_sockopt_ttl();
    test_sockopt_read_timeout();
    test_sockopt_write_timeout();
}

fn test_create_ipv4() {
    let _socket_ipv4 = UdpSocket::bind("127.0.0.1:0").unwrap();
}

fn test_create_ipv6() {
    let _socket_ipv6 = UdpSocket::bind("[::1]:0").unwrap();
}

/// Test setting the default destination address using `connect`.
fn test_connect() {
    let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    socket.connect("127.0.1.1:8080").unwrap()
}

// Test setting and reading the TTL socket option.
fn test_sockopt_ttl() {
    let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    socket.set_ttl(16).unwrap();
    assert_eq!(socket.ttl().unwrap(), 16);
}

/// Test setting and reading the SNDTIMEO socket option.
/// This also tests that a read won't block indefinitely
/// when the read timeout is set to [`Some`] duration.
fn test_sockopt_read_timeout() {
    let socket = UdpSocket::bind("127.0.0.1:0").unwrap();

    // By default, reads on blocking sockets should block indefinitely.
    assert_eq!(socket.read_timeout().unwrap(), None);

    let short_read_timeout = Some(Duration::from_millis(40));
    socket.set_read_timeout(short_read_timeout).unwrap();
    assert_eq!(socket.read_timeout().unwrap(), short_read_timeout);

    /*
    TODO: Once UDP sockets support reading.

    let mut buffer = [0u8; 128];
    // This should not block indefinitely and instead return EAGAIN/EWOULDBLOCK.
    let err = socket.read(&mut buffer).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::WouldBlock);
    */
}

/// Test setting and reading the RCVTIMEO socket option.
/// This also tests that a write won't block indefinitely when
/// the write timeout is set to [`Some`] duration.
fn test_sockopt_write_timeout() {
    let socket = UdpSocket::bind("127.0.0.1:0").unwrap();

    // By default, writes on blocking sockets should block indefinitely.
    assert_eq!(socket.write_timeout().unwrap(), None);

    let short_write_timeout = Some(Duration::from_millis(40));
    socket.set_write_timeout(short_write_timeout).unwrap();
    assert_eq!(socket.write_timeout().unwrap(), short_write_timeout);

    /*
    TODO: Once UDP sockets support writing.

    let fill_buffer = [1u8; 1024];
    loop {
        match socket.write_all(&fill_buffer) {
            Ok(_) => { /* continue to fill up buffer */ }
            // When we get an EAGAIN/EWOULDBLOCK when writing into a blocking socket,
            // we know it's because of the write timeout exceeding because the write
            // buffer is full.
            Err(err) if err.kind() == ErrorKind::WouldBlock => break,
            Err(err) => panic!("unexpected error whilst filling up buffer: {err}"),
        }
    }
    */
}
