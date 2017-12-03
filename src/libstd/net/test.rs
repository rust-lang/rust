// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(warnings)] // not used on emscripten

use env;
use net::{SocketAddr, SocketAddrV4, SocketAddrV6, Ipv4Addr, Ipv6Addr,
          ToSocketAddrs, TcpListener};
use sync::atomic::{AtomicUsize, Ordering};
use sync::{Once, ONCE_INIT};
use io::ErrorKind;

// If the computer does not have a working IPv4 or v6 loopback address,
// all of the v4 or v6 tests, respectively, should be skipped.
// We let the user override our attempt to detect this automatically
// by setting an environment variable.

enum TestAddrFamily { No, Yes, Maybe }
fn test_addr_family_env(var: &'static str) -> TestAddrFamily {
    use self::TestAddrFamily::*;

    match env::var_os(var) {
        None      => Maybe,
        Some(ref s) => {
            if s == "1" {
                Yes
            } else if s == "0" {
                No
            } else {
                eprintln!("warning: ignoring strange value for '{}'", var);
                eprintln!("warning: understood values are 0 and 1");
                Maybe
            }
        }
    }
}

fn test_addr_family_auto(addr: SocketAddr) -> bool {
    // An address family is assumed to be usable if it is possible to
    // bind an OS-assigned port on that address family's loopback
    // address, and to not be usable if the attempt fails with
    // EADDRNOTAVAIL.  Other errors are treated as fatal.  (Caller is
    // responsible for supplying an appropriate SocketAddr.)
    match TcpListener::bind(addr) {
        Ok(..) => true,
        Err(e) => {
            assert!(e.kind() == ErrorKind::AddrNotAvailable,
                    "address family probe failure: {}", e);
            false
        }
    }
}

pub fn test_ipv4_p() -> bool {
    use self::TestAddrFamily::*;
    static mut TEST_IPV4: bool = false;
    static TEST_IPV4_INIT: Once = ONCE_INIT;

    TEST_IPV4_INIT.call_once(|| {
        let test_ipv4 = match test_addr_family_env("RUST_TEST_IPV4") {
            Yes    => true,
            No     => false,
            Maybe  => test_addr_family_auto(sa4(
                Ipv4Addr::new(127, 0, 0, 1), 0))
        };
        unsafe { TEST_IPV4 = test_ipv4 }
    });
    unsafe { TEST_IPV4 }
}

pub fn test_ipv6_p() -> bool {
    use self::TestAddrFamily::*;
    static mut TEST_IPV6: bool = false;
    static TEST_IPV6_INIT: Once = ONCE_INIT;

    TEST_IPV6_INIT.call_once(|| {
        let test_ipv6 = match test_addr_family_env("RUST_TEST_IPV6") {
            Yes    => true,
            No     => false,
            Maybe  => test_addr_family_auto(sa6(
                Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1), 0))
        };
        unsafe { TEST_IPV6 = test_ipv6 }
    });
    unsafe { TEST_IPV6 }
}

// The bots run multiple builds at the same time, and these builds
// all want to use ports. This function figures out which workspace
// it is running in and assigns a port range based on it.
fn base_port() -> u16 {
    static mut BASE_PORT: u16 = 0;
    static BASE_PORT_INIT: Once = ONCE_INIT;

    BASE_PORT_INIT.call_once(|| {
        let cwd_p = env::current_dir().unwrap();
        let cwd = cwd_p.to_str().unwrap();
        let dirs = ["32-opt", "32-nopt",
                    "musl-64-opt", "cross-opt",
                    "64-opt", "64-nopt", "64-opt-vg", "64-debug-opt",
                    "all-opt", "snap3", "dist"];
        let base_port = dirs.iter().enumerate().find(|&(_, dir)| {
            cwd.contains(dir)
        }).map(|p| p.0).unwrap_or(0) as u16 * 1000 + 19600;
        unsafe { BASE_PORT = base_port }
    });
    unsafe { BASE_PORT }
}

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
