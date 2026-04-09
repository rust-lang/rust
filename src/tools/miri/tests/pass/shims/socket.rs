//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_create_ipv4_listener();
    test_create_ipv6_listener();
    test_accept_and_connect();
    test_read_write();
    test_peek();
    test_peer_addr();
}

fn test_create_ipv4_listener() {
    let _listener_ipv4 = TcpListener::bind("127.0.0.1:0").unwrap();
}

fn test_create_ipv6_listener() {
    let _listener_ipv6 = TcpListener::bind("[::1]:0").unwrap();
}

/// Try to connect to a TCP listener running in a separate thread and
/// accepting connections.
fn test_accept_and_connect() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    // Get local address with randomized port to know where
    // we need to connect to.
    let address = listener.local_addr().unwrap();

    let handle = thread::spawn(move || {
        let (_stream, _addr) = listener.accept().unwrap();
    });

    let _stream = TcpStream::connect(address).unwrap();

    handle.join().unwrap();
}

/// Test reading and writing into two connected sockets and ensuring
/// that the other side receives the correct bytes.
fn test_read_write() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    // Get local address with randomized port to know where
    // we need to connect to.
    let address = listener.local_addr().unwrap();

    let handle = thread::spawn(move || {
        let (mut stream, _addr) = listener.accept().unwrap();
        stream.write_all(TEST_BYTES).unwrap();
    });

    let mut stream = TcpStream::connect(address).unwrap();

    let mut buffer = [0; TEST_BYTES.len()];
    stream.read_exact(&mut buffer).unwrap();
    assert_eq!(&buffer, TEST_BYTES);

    handle.join().unwrap();
}

/// Test that peeking on a receiving socket doesn't remove the bytes
/// from the queue.
fn test_peek() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    // Get local address with randomized port to know where
    // we need to connect to.
    let address = listener.local_addr().unwrap();

    let handle = thread::spawn(move || {
        let (mut stream, _addr) = listener.accept().unwrap();
        stream.write_all(TEST_BYTES).unwrap();
    });

    let mut stream = TcpStream::connect(address).unwrap();

    let mut buffer = [0; TEST_BYTES.len()];
    let bytes_peeked = stream.peek(&mut buffer).unwrap();
    // Since we have short accesses, and there is nothing like an "peek exact",
    // we can only ensure that we peeked at most as many bytes as we wrote into
    // the buffer and that the peeked bytes match the ones we wrote.
    assert!(bytes_peeked <= TEST_BYTES.len());
    assert_eq!(&buffer[0..bytes_peeked], &TEST_BYTES[0..bytes_peeked]);

    // Since the peek shouldn't remove the bytes from the queue,
    // we should be able to read them again.
    let mut buffer = [0; TEST_BYTES.len()];
    stream.read_exact(&mut buffer).unwrap();
    assert_eq!(&buffer, TEST_BYTES);

    handle.join().unwrap();
}

/// Test whether the [`TcpStream::peer_addr`] of a connected socket
/// is the same address as the one the stream was connected to.
fn test_peer_addr() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    // Get local address with randomized port to know where
    // we need to connect to.
    let address = listener.local_addr().unwrap();

    let handle = thread::spawn(move || {
        let (_stream, _addr) = listener.accept().unwrap();
    });

    let stream = TcpStream::connect(address).unwrap();
    let peer_addr = stream.peer_addr().unwrap();
    assert_eq!(address, peer_addr);

    handle.join().unwrap();
}
