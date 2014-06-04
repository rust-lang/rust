// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Networking I/O

use rt::rtio;
use self::ip::{Ipv4Addr, Ipv6Addr, IpAddr};

pub use self::addrinfo::get_host_addresses;

pub mod addrinfo;
pub mod tcp;
pub mod udp;
pub mod ip;
// FIXME(#12093) - this should not be called unix
pub mod unix;

fn to_rtio(ip: IpAddr) -> rtio::IpAddr {
    match ip {
        Ipv4Addr(a, b, c, d) => rtio::Ipv4Addr(a, b, c, d),
        Ipv6Addr(a, b, c, d, e, f, g, h) => {
            rtio::Ipv6Addr(a, b, c, d, e, f, g, h)
        }
    }
}

fn from_rtio(ip: rtio::IpAddr) -> IpAddr {
    match ip {
        rtio::Ipv4Addr(a, b, c, d) => Ipv4Addr(a, b, c, d),
        rtio::Ipv6Addr(a, b, c, d, e, f, g, h) => {
            Ipv6Addr(a, b, c, d, e, f, g, h)
        }
    }
}
