//@ignore-target: windows # No socket support on Windows
//@compile-flags: -Zmiri-disable-isolation
//@run-native

use std::io::ErrorKind;
use std::net::UdpSocket;
use std::thread;
use std::time::{Duration, Instant};

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_create_ipv4();
    test_create_ipv6();
    test_connect();
    test_read_write();
    test_peek();
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

/// Test reading and writing into two connected sockets and ensuring
/// that the other side receives the correct bytes.
fn test_read_write() {
    let tx_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    let rx_socket = UdpSocket::bind("127.0.0.1:0").unwrap();

    // Get local address with randomized port to know where we
    // need to send the data to.
    let address = rx_socket.local_addr().unwrap();

    // Set the default destination address of the transmitting socket
    // to the address of the receiving socket.
    tx_socket.connect(address).unwrap();

    let handle = thread::spawn(move || {
        let bytes_sent = tx_socket.send(TEST_BYTES).unwrap();
        assert_eq!(bytes_sent, TEST_BYTES.len());
    });

    let mut buffer = [0; TEST_BYTES.len()];
    let bytes_received = rx_socket.recv(&mut buffer).unwrap();
    assert_eq!(bytes_received, TEST_BYTES.len());
    assert_eq!(&buffer, TEST_BYTES);

    handle.join().unwrap();
}

/// Test that peeking on a receiving socket doesn't remove the bytes
/// from the queue.
fn test_peek() {
    let tx_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    let rx_socket = UdpSocket::bind("127.0.0.1:0").unwrap();

    // Get local address with randomized port to know where we
    // need to send the data to.
    let address = rx_socket.local_addr().unwrap();

    // Set the default destination address of the transmitting socket
    // to the address of the receiving socket.
    tx_socket.connect(address).unwrap();

    let handle = thread::spawn(move || {
        let bytes_sent = tx_socket.send(TEST_BYTES).unwrap();
        assert_eq!(bytes_sent, TEST_BYTES.len());
    });

    let mut buffer = [0; TEST_BYTES.len()];
    let bytes_peeked = rx_socket.peek(&mut buffer).unwrap();
    assert_eq!(bytes_peeked, TEST_BYTES.len());
    assert_eq!(&buffer, &TEST_BYTES);

    // Since the peek shouldn't remove the bytes from the queue,
    // we should be able to read them again.
    let mut buffer = [0; TEST_BYTES.len()];
    let bytes_received = rx_socket.recv(&mut buffer).unwrap();
    assert_eq!(bytes_received, TEST_BYTES.len());
    assert_eq!(&buffer, TEST_BYTES);

    handle.join().unwrap();
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

    let short_read_timeout = Duration::from_millis(40);
    socket.set_read_timeout(Some(short_read_timeout)).unwrap();
    assert_eq!(socket.read_timeout().unwrap(), Some(short_read_timeout));

    let mut buffer = [0u8; 128];
    let before = Instant::now();
    // This should not block indefinitely and instead return EAGAIN/EWOULDBLOCK.
    let err = socket.recv(&mut buffer).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::WouldBlock);
    // Ensure that we blocked at least for `short_read_timeout`. The upper bound on the
    // blocking time is pessimistic to account for slow CI runners.
    let duration = before.elapsed();
    assert!(
        duration >= short_read_timeout && duration < short_read_timeout * 20,
        "operation blocked for {duration:?}"
    );
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

    // We cannot test "filling up" the UDP socket buffer since UDP is unreliable.
    // This means the kernel can report to us that it sent the entire datagram but
    // on the receiving end it just gets discarded should the buffer be full without
    // notifying the sender.
}
