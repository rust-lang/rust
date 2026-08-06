//@ignore-target: windows # No socket support on Windows
//@compile-flags: -Zmiri-disable-isolation
//@run-native

use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::thread;
use std::time::{Duration, Instant};

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_create_ipv4_listener();
    test_create_ipv6_listener();
    test_accept_and_connect();
    test_connect_with_timeout();
    test_connect_with_timeout_error();
    test_read_write();
    test_peek();
    test_peer_addr();
    test_shutdown();
    test_sockopt_ttl();
    test_sockopt_nodelay();
    test_sockopt_read_timeout();
    test_sockopt_write_timeout();
}

fn test_create_ipv4_listener() {
    let _listener_ipv4 = TcpListener::bind("127.0.0.1:0").unwrap();
}

fn test_create_ipv6_listener() {
    let _listener_ipv6 = TcpListener::bind("[::1]:0").unwrap();
}

/// Try to connect to a TCP listener and accepting connections.
fn test_accept_and_connect() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    // Get local address with randomized port to know where
    // we need to connect to.
    let address = listener.local_addr().unwrap();

    let _stream = TcpStream::connect(address).unwrap();
    let (_other_stream, _addr) = listener.accept().unwrap();
}

/// Test connecting a socket with a timeout. The timeout
/// shouldn't expire because a listener is listening on
/// the specified address.
fn test_connect_with_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    // Get local address with randomized port to know where
    // we need to connect to.
    let address = listener.local_addr().unwrap();

    let _stream = TcpStream::connect_timeout(&address, Duration::from_millis(50)).unwrap();
    let (_other_stream, _addr) = listener.accept().unwrap();
}

/// Test connecting a socket with a timeout which expires
/// because we attempt to connect to an address where nothing
/// is listening.
fn test_connect_with_timeout_error() {
    // We cannot attempt to connect to a localhost address because
    // it could be the case that a socket from another test is
    // currently listening on `localhost:12345` because we bind to
    // random ports everywhere. For `192.0.2.1` we know that nothing is
    // listening because it's a blackhole address:
    // <https://www.rfc-editor.org/rfc/rfc5737>
    // The port `12345` is just a random non-zero port.
    let address = "192.0.2.1:12345".parse::<SocketAddr>().unwrap();
    let timeout = Duration::from_millis(50);

    let before = Instant::now();
    let err = TcpStream::connect_timeout(&address, timeout).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TimedOut);
    assert!(before.elapsed() >= timeout);
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
    listener.set_ttl(16).unwrap();
    assert_eq!(listener.ttl().unwrap(), 16);
}

/// Test setting and reading the TCP nodelay socket option.
fn test_sockopt_nodelay() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();

    let stream = TcpStream::connect(address).unwrap();
    let _other_end = listener.accept().unwrap();

    stream.set_nodelay(true).unwrap();
    assert_eq!(stream.nodelay().unwrap(), true);
    stream.set_nodelay(false).unwrap();
    assert_eq!(stream.nodelay().unwrap(), false);
}

/// Test setting and reading the SNDTIMEO socket option.
/// This also tests that a read won't block indefinitely
/// when the read timeout is set to [`Some`] duration.
fn test_sockopt_read_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();

    let mut stream = TcpStream::connect(address).unwrap();
    let _other_end = listener.accept().unwrap();

    // By default, reads on blocking sockets should block indefinitely.
    assert_eq!(stream.read_timeout().unwrap(), None);

    let short_read_timeout = Some(Duration::from_millis(40));
    stream.set_read_timeout(short_read_timeout).unwrap();
    assert_eq!(stream.read_timeout().unwrap(), short_read_timeout);

    let mut buffer = [0u8; 128];
    // This should not block indefinitely and instead return EAGAIN/EWOULDBLOCK.
    let err = stream.read(&mut buffer).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::WouldBlock);
}

/// Test setting and reading the RCVTIMEO socket option.
/// This also tests that a write won't block indefinitely when
/// the write timeout is set to [`Some`] duration.
fn test_sockopt_write_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();

    let mut stream = TcpStream::connect(address).unwrap();
    let _other_end = listener.accept().unwrap();

    // By default, writes on blocking sockets should block indefinitely.
    assert_eq!(stream.write_timeout().unwrap(), None);

    let short_write_timeout = Some(Duration::from_millis(40));
    stream.set_write_timeout(short_write_timeout).unwrap();
    assert_eq!(stream.write_timeout().unwrap(), short_write_timeout);

    let fill_buffer = [1u8; 1024];
    loop {
        match stream.write_all(&fill_buffer) {
            Ok(_) => { /* continue to fill up buffer */ }
            // When we get an EAGAIN/EWOULDBLOCK when writing into a blocking socket,
            // we know it's because of the write timeout exceeding because the write
            // buffer is full.
            Err(err) if err.kind() == ErrorKind::WouldBlock => break,
            Err(err) => panic!("unexpected error whilst filling up buffer: {err}"),
        }
    }
}
