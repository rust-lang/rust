//@ignore-target: windows
//@compile-flags: -Zmiri-fixed-schedule
//@run-native

use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::UnixStream;
use std::thread;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn yield_now() {
    if cfg!(miri) {
        // We control the scheduler so we know this is enough.
        thread::yield_now();
    } else {
        // Without scheduler control, we hope that sleeping a bit is enough.
        thread::sleep(std::time::Duration::from_millis(20));
    }
}

fn main() {
    test_send_recv();
    test_send_recv_blocking();
    test_change_blocking_mode();
}

/// Test sending and receiving data on a pair of sockets.
fn test_send_recv() {
    let (mut sock1, mut sock2) = UnixStream::pair().unwrap();
    sock1.write_all(TEST_BYTES).unwrap();
    drop(sock1); // close it so that the other side gets an EOF.

    let mut response = Vec::new();
    sock2.read_to_end(&mut response).unwrap();
    assert_eq!(response, TEST_BYTES);
}

fn test_send_recv_blocking() {
    let (mut sock1, mut sock2) = UnixStream::pair().unwrap();
    sock1.write_all(TEST_BYTES).unwrap();
    let t = thread::spawn(move || {
        let mut response = Vec::new();
        sock2.read_to_end(&mut response).unwrap();
        response
    });
    yield_now(); // let the other thread block just before we close our side
    drop(sock1); // close it so that the other side gets an EOF.
    let data = t.join().unwrap();
    assert_eq!(data, TEST_BYTES);
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
