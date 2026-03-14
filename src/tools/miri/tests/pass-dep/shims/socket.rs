//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation

use std::net::TcpListener;

fn main() {
    create_ipv4_listener();
    create_ipv6_listener();
}

fn create_ipv4_listener() {
    let _listener_ipv4 = TcpListener::bind("127.0.0.1:0").unwrap();
}

fn create_ipv6_listener() {
    let _listener_ipv6 = TcpListener::bind("[::1]:0").unwrap();
}
