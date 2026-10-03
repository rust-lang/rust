//@ignore-target: windows # No socket support on Windows
//@compile-flags: -Zmiri-disable-isolation
//@run-native

use std::net::UdpSocket;

fn main() {
    test_create_ipv4();
    test_create_ipv6();
    test_connect();
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
