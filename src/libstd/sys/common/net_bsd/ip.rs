// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cmp::Ordering;
use hash;
use libc;
use super::{hton, ntoh};

#[derive(Copy)]
pub struct IpAddrV4(libc::in_addr);

#[derive(Copy)]
pub struct IpAddrV6(libc::in6_addr);

impl_inner!(IpAddrV4(libc::in_addr));
impl_inner!(IpAddrV6(libc::in6_addr));

impl IpAddrV4 {
    pub fn new(a: u8, b: u8, c: u8, d: u8) -> IpAddrV4 {
        IpAddrV4(
            libc::in_addr {
                s_addr: hton(((a as u32) << 24) |
                             ((b as u32) << 16) |
                             ((c as u32) <<  8) |
                              (d as u32)),
            }
        )
    }

    pub fn octets(&self) -> [u8; 4] {
        let bits = ntoh(self.0.s_addr);
        [(bits >> 24) as u8, (bits >> 16) as u8, (bits >> 8) as u8, bits as u8]
    }
}

impl Clone for IpAddrV4 {
    fn clone(&self) -> IpAddrV4 { *self }
}

impl PartialEq for IpAddrV4 {
    fn eq(&self, other: &IpAddrV4) -> bool {
        self.0.s_addr == other.0.s_addr
    }
}

impl Eq for IpAddrV4 {}

impl hash::Hash for IpAddrV4 {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        self.0.s_addr.hash(s)
    }
}

impl PartialOrd for IpAddrV4 {
    fn partial_cmp(&self, other: &IpAddrV4) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IpAddrV4 {
    fn cmp(&self, other: &IpAddrV4) -> Ordering {
        self.0.s_addr.cmp(&other.0.s_addr)
    }
}

impl IpAddrV6 {
    pub fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16,
               h: u16) -> IpAddrV6 {
        IpAddrV6(
            libc::in6_addr {
                s6_addr: [hton(a), hton(b), hton(c), hton(d),
                          hton(e), hton(f), hton(g), hton(h)]
            }
        )
    }

    /// Returns the eight 16-bit segments that make up this address.
    pub fn segments(&self) -> [u16; 8] {
        [ntoh(self.0.s6_addr[0]),
         ntoh(self.0.s6_addr[1]),
         ntoh(self.0.s6_addr[2]),
         ntoh(self.0.s6_addr[3]),
         ntoh(self.0.s6_addr[4]),
         ntoh(self.0.s6_addr[5]),
         ntoh(self.0.s6_addr[6]),
         ntoh(self.0.s6_addr[7])]
    }
}

impl Clone for IpAddrV6 {
    fn clone(&self) -> IpAddrV6 { *self }
}

impl PartialEq for IpAddrV6 {
    fn eq(&self, other: &IpAddrV6) -> bool {
        self.0.s6_addr == other.0.s6_addr
    }
}

impl Eq for IpAddrV6 {}

impl hash::Hash for IpAddrV6 {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        self.0.s6_addr.hash(s)
    }
}

impl PartialOrd for IpAddrV6 {
    fn partial_cmp(&self, other: &IpAddrV6) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IpAddrV6 {
    fn cmp(&self, other: &IpAddrV6) -> Ordering {
        self.0.s6_addr.cmp(&other.0.s6_addr)
    }
}
