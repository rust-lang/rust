//@ignore-target: windows # No socket support on Windows
//@compile-flags: -Zmiri-disable-isolation
//@run-native

use std::net::UdpSocket;

fn main() {
    test_create_ipv4();
    test_create_ipv6();
}

fn test_create_ipv4() {
    let _socket_ipv4 = UdpSocket::bind("127.0.0.1:0").unwrap();
}

fn test_create_ipv6() {
    let _socket_ipv6 = UdpSocket::bind("[::1]:0").unwrap();
}
