// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_escape];

use os;
use prelude::*;
use rand;
use rand::Rng;
use std::io::net::ip::*;
use sync::atomics::{AtomicUint, INIT_ATOMIC_UINT, Relaxed};

macro_rules! iotest (
    { fn $name:ident() $b:block } => (
        mod $name {
            #[allow(unused_imports)];

            use super::super::*;
            use super::*;
            use io;
            use prelude::*;
            use io::*;
            use io::fs::*;
            use io::net::tcp::*;
            use io::net::ip::*;
            use io::net::udp::*;
            use io::net::unix::*;
            use str;
            use util;

            fn f() $b

            #[test] fn green() { f() }
            #[test] fn native() {
                use native;
                let (p, c) = Chan::new();
                do native::task::spawn { c.send(f()) }
                p.recv();
            }
        }
    )
)

/// Get a port number, starting at 9600, for use in tests
pub fn next_test_port() -> u16 {
    static mut next_offset: AtomicUint = INIT_ATOMIC_UINT;
    unsafe {
        base_port() + next_offset.fetch_add(1, Relaxed) as u16
    }
}

/// Get a temporary path which could be the location of a unix socket
pub fn next_test_unix() -> Path {
    if cfg!(unix) {
        os::tmpdir().join(rand::task_rng().gen_ascii_str(20))
    } else {
        Path::new(r"\\.\pipe\" + rand::task_rng().gen_ascii_str(20))
    }
}

/// Get a unique IPv4 localhost:port pair starting at 9600
pub fn next_test_ip4() -> SocketAddr {
    SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: next_test_port() }
}

/// Get a unique IPv6 localhost:port pair starting at 9600
pub fn next_test_ip6() -> SocketAddr {
    SocketAddr { ip: Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1), port: next_test_port() }
}

/*
XXX: Welcome to MegaHack City.

The bots run multiple builds at the same time, and these builds
all want to use ports. This function figures out which workspace
it is running in and assigns a port range based on it.
*/
fn base_port() -> u16 {

    let base = 9600u16;
    let range = 1000u16;

    let bases = [
        ("32-opt", base + range * 1),
        ("32-noopt", base + range * 2),
        ("64-opt", base + range * 3),
        ("64-noopt", base + range * 4),
        ("64-opt-vg", base + range * 5),
        ("all-opt", base + range * 6),
        ("snap3", base + range * 7),
        ("dist", base + range * 8)
    ];

    // FIXME (#9639): This needs to handle non-utf8 paths
    let path = os::getcwd();
    let path_s = path.as_str().unwrap();

    let mut final_base = base;

    for &(dir, base) in bases.iter() {
        if path_s.contains(dir) {
            final_base = base;
            break;
        }
    }

    return final_base;
}
