//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation

use std::net::{TcpListener, TcpStream};
use std::thread;

fn main() {
    test_create_ipv4_listener();
    test_create_ipv6_listener();
    test_accept_and_connect();
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
