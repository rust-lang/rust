//@ignore-target: windows

use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::UnixStream;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_send_recv();
    test_change_blocking_mode();
}

/// Test sending and receiving data on a pair of sockets.
fn test_send_recv() {
    let (mut sock1, mut sock2) = UnixStream::pair().unwrap();
    sock1.write_all(TEST_BYTES).unwrap();
    drop(sock1); // close it so that the other side gets an EOF.

    let mut response = String::new();
    sock2.read_to_string(&mut response).unwrap();
    assert_eq!(response.as_bytes(), TEST_BYTES);
}

/// Test changing the blocking mode of a socket from a socket pair.
fn test_change_blocking_mode() {
    let (mut sock1, _sock2) = UnixStream::pair().unwrap();
    // Change the blocking mode of one of the sockets.
    sock1.set_nonblocking(true).unwrap();

    let mut buffer = [0u8; 1];
    // Reading from the socket should block because the peer didn't write anything.
    let err = sock1.read(&mut buffer).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::WouldBlock);
}
