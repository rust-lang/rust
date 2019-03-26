#![allow(warnings)] // not used on emscripten

use crate::env;
use crate::net::{SocketAddr, SocketAddrV4, SocketAddrV6, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use crate::sync::atomic::{AtomicUsize, Ordering};

static PORT: AtomicUsize = AtomicUsize::new(0);

pub fn next_test_ip4() -> SocketAddr {
    let port = PORT.fetch_add(1, Ordering::SeqCst) as u16 + base_port();
    SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), port))
}

pub fn next_test_ip6() -> SocketAddr {
    let port = PORT.fetch_add(1, Ordering::SeqCst) as u16 + base_port();
    SocketAddr::V6(SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1),
                                     port, 0, 0))
}

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

// The bots run multiple builds at the same time, and these builds
// all want to use ports. This function figures out which workspace
// it is running in and assigns a port range based on it.
fn base_port() -> u16 {
    let cwd = if cfg!(target_env = "sgx") {
        String::from("sgx")
    } else {
        env::current_dir().unwrap().into_os_string().into_string().unwrap()
    };
    let dirs = ["32-opt", "32-nopt",
                "musl-64-opt", "cross-opt",
                "64-opt", "64-nopt", "64-opt-vg", "64-debug-opt",
                "all-opt", "snap3", "dist", "sgx"];
    dirs.iter().enumerate().find(|&(_, dir)| {
        cwd.contains(dir)
    }).map(|p| p.0).unwrap_or(0) as u16 * 1000 + 19600
}
