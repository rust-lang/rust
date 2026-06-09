#![allow(warnings)] // not used on emscripten

use crate::env;
use crate::net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6, ToSocketAddrs};
use crate::sync::atomic::{AtomicUsize, Ordering};

/// A localhost address whose port will be picked automatically by the OS.
pub const LOCALHOST_IP4: SocketAddr =
    SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 0));
/// A localhost address whose port will be picked automatically by the OS.
pub const LOCALHOST_IP6: SocketAddr =
    SocketAddr::V6(SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1), 0, 0, 0));

pub fn sa4(a: Ipv4Addr, p: u16) -> SocketAddr {
    SocketAddr::V4(SocketAddrV4::new(a, p))
}

pub fn sa6(a: Ipv6Addr, p: u16) -> SocketAddr {
    SocketAddr::V6(SocketAddrV6::new(a, p, 0, 0))
}

pub fn tsa<A: ToSocketAddrs>(a: A) -> Result<Vec<SocketAddr>, String> {
    match a.to_socket_addrs() {
        Ok(a) => Ok(a.collect()),
        Err(e) => Err(e.to_string()),
    }
}

pub fn compare_ignore_zoneid(a: &SocketAddr, b: &SocketAddr) -> bool {
    match (a, b) {
        (SocketAddr::V6(a), SocketAddr::V6(b)) => {
            a.ip().segments() == b.ip().segments()
                && a.flowinfo() == b.flowinfo()
                && a.port() == b.port()
        }
        _ => a == b,
    }
}
