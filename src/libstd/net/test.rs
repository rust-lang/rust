// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use env;
use net::{SocketAddr, SocketAddrV4, SocketAddrV6, Ipv4Addr, Ipv6Addr};
use sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};

static PORT: AtomicUsize = ATOMIC_USIZE_INIT;

pub fn next_test_ip4() -> SocketAddr {
    let port = PORT.fetch_add(1, Ordering::SeqCst) as u16 + base_port();
    SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), port))
}

pub fn next_test_ip6() -> SocketAddr {
    let port = PORT.fetch_add(1, Ordering::SeqCst) as u16 + base_port();
    SocketAddr::V6(SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1),
                                     port, 0, 0))
}

// The bots run multiple builds at the same time, and these builds
// all want to use ports. This function figures out which workspace
// it is running in and assigns a port range based on it.
fn base_port() -> u16 {
    let cwd = env::current_dir().unwrap();
    let dirs = ["32-opt", "32-nopt", "64-opt", "64-nopt", "64-opt-vg",
                "all-opt", "snap3", "dist"];
    dirs.iter().enumerate().find(|&(_, dir)| {
        cwd.to_str().unwrap().contains(dir)
    }).map(|p| p.0).unwrap_or(0) as u16 * 1000 + 19600
}
