//@ignore-target: windows # No socket support on Windows
//@compile-flags: -Zmiri-disable-isolation

use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::thread;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_create_ipv4_listener();
    test_create_ipv6_listener();
    test_accept_and_connect();
    test_read_write();
    test_peek();
    test_peer_addr();
    test_shutdown();
    test_sockopt_ttl();
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

/// Test shutting down TCP streams.
fn test_shutdown() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    // Get local address with randomized port to know where
    // we need to connect to.
    let address = listener.local_addr().unwrap();

    // Start server thread.
    let handle = thread::spawn(move || {
        let (stream, _addr) = listener.accept().unwrap();
        // Return stream from thread such that it doesn't get dropped too early.
        stream
    });

    let mut byte = [0u8];
    let mut stream = TcpStream::connect(address).unwrap();
    let mut stream_clone = stream.try_clone().unwrap();

    // Closing should prevent reads/writes.
    stream.shutdown(Shutdown::Write).unwrap();
    stream.write(&[0]).unwrap_err();
    stream.shutdown(Shutdown::Read).unwrap();
    assert_eq!(stream.read(&mut byte).unwrap(), 0);

    // Closing should affect previously cloned handles.
    let err = stream_clone.write(&[0]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::BrokenPipe);
    assert_eq!(stream_clone.read(&mut byte).unwrap(), 0);

    // Closing should affect newly cloned handles.
    let mut stream_other_clone = stream.try_clone().unwrap();
    let err = stream_other_clone.write(&[0]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::BrokenPipe);
    assert_eq!(stream_other_clone.read(&mut byte).unwrap(), 0);

    let _stream = handle.join().unwrap();
}

/// Test setting and reading the TTL socket option.
fn test_sockopt_ttl() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.ttl().unwrap();

    // TODO: Once we support setting the TTL we should also test it here.
}
