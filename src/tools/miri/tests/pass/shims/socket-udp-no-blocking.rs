//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation

use std::io::ErrorKind;
use std::net::UdpSocket;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_send_recv_nonblock();
}

/// Test sending bytes into one UDP socket and receiving them from another UDP socket
/// without blocking.
fn test_send_recv_nonblock() {
    let sock_rx = UdpSocket::bind("127.0.0.1:0").unwrap();
    sock_rx.set_nonblocking(true).unwrap();

    let sock_tx = UdpSocket::bind("127.0.0.1:0").unwrap();
    sock_tx.set_nonblocking(true).unwrap();

    // Get local address with randomized port to know where
    // we need to send the bytes to.
    let address_rx = sock_rx.local_addr().unwrap();

    // Set the default destination address of the transmitting UDP socket
    // to the address where the receiving UDP socket is bound to.
    sock_tx.connect(address_rx).unwrap();

    let mut buffer = [0u8; 1024];
    // Receiving when no data is available should fail with EWOULDBLOCK.
    let err = sock_rx.recv(&mut buffer).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::WouldBlock);

    let bytes_sent = sock_tx.send(TEST_BYTES).unwrap();
    assert_eq!(bytes_sent, TEST_BYTES.len());

    let bytes_received = sock_rx.recv(&mut buffer).unwrap();
    assert_eq!(bytes_received, TEST_BYTES.len());
    assert_eq!(&buffer[0..bytes_received], TEST_BYTES);
}
