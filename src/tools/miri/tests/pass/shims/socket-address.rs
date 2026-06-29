//@ignore-target: windows # No socket address support on Windows
//@compile-flags: -Zmiri-disable-isolation
//@normalize-stderr-test: "address resolution failed: .*" -> "address resolution failed: $$MSG"

use std::net::{
    IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6, TcpListener, ToSocketAddrs,
};

fn main() {
    test_address_resolution();
}

/// Test getting a socket address from a hostname and a port.
fn test_address_resolution() {
    let listener = TcpListener::bind("localhost:0").unwrap();
    let address = listener.local_addr().unwrap();
    match address.ip() {
        IpAddr::V4(addr) => assert_eq!(addr, Ipv4Addr::LOCALHOST),
        IpAddr::V6(addr) => assert_eq!(addr, Ipv6Addr::LOCALHOST),
    }

    let addr_str = "localhost:8888";
    let mut addr_count = 0;
    for addr in addr_str.to_socket_addrs().unwrap() {
        addr_count += 1;
        match addr {
            SocketAddr::V4(addr) => assert_eq!(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 8888), addr),
            SocketAddr::V6(addr) =>
                assert_eq!(SocketAddrV6::new(Ipv6Addr::LOCALHOST, 8888, 0, 0), addr),
        }
    }
    // We expect an IPv4 and an IPv6 address.
    assert!(addr_count == 2);

    // Resolving an invalid name should error. Needs the port to even hit `getaddrinfo`.
    let addr_str = "this-is-not-a-valid-address:80";
    addr_str.to_socket_addrs().unwrap_err();
}
