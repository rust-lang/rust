// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use inner::prelude::*;
use core::cmp::Ordering;
use core::hash;
use core::fmt;
use libc;
use super::{hton, ntoh};
use net as sys;

#[derive(Copy)]
pub struct Ipv4Addr {
    inner: libc::in_addr,
}

#[derive(Copy)]
pub struct Ipv6Addr {
    inner: libc::in6_addr,
}

impl sys::AddrV4 for Ipv4Addr {
    fn new(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr {
        Ipv4Addr {
            inner: libc::in_addr {
                s_addr: hton(((a as u32) << 24) |
                             ((b as u32) << 16) |
                             ((c as u32) <<  8) |
                              (d as u32)),
            }
        }
    }

    fn octets(&self) -> [u8; 4] {
        let bits = ntoh(self.inner.s_addr);
        [(bits >> 24) as u8, (bits >> 16) as u8, (bits >> 8) as u8, bits as u8]
    }
}

impl AsInner<libc::in_addr> for Ipv4Addr {
    fn as_inner(&self) -> &libc::in_addr {
        &self.inner
    }
}

impl IntoInner<libc::in_addr> for Ipv4Addr {
    fn into_inner(self) -> libc::in_addr {
        self.inner
    }
}

impl FromInner<libc::in_addr> for Ipv4Addr {
    fn from_inner(inner: libc::in_addr) -> Self {
        Ipv4Addr { inner: inner }
    }
}

impl Clone for Ipv4Addr {
    fn clone(&self) -> Ipv4Addr { *self }
}

impl PartialEq for Ipv4Addr {
    fn eq(&self, other: &Ipv4Addr) -> bool {
        self.inner.s_addr == other.inner.s_addr
    }
}

impl Eq for Ipv4Addr {}

impl hash::Hash for Ipv4Addr {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        self.inner.s_addr.hash(s)
    }
}

impl PartialOrd for Ipv4Addr {
    fn partial_cmp(&self, other: &Ipv4Addr) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ipv4Addr {
    fn cmp(&self, other: &Ipv4Addr) -> Ordering {
        self.inner.s_addr.cmp(&other.inner.s_addr)
    }
}

impl sys::AddrV6 for Ipv6Addr {
    fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16,
               h: u16) -> Ipv6Addr {
        Ipv6Addr {
            inner: libc::in6_addr {
                s6_addr: [hton(a), hton(b), hton(c), hton(d),
                          hton(e), hton(f), hton(g), hton(h)]
            }
        }
    }

    /// Returns the eight 16-bit segments that make up this address.
    fn segments(&self) -> [u16; 8] {
        [ntoh(self.inner.s6_addr[0]),
         ntoh(self.inner.s6_addr[1]),
         ntoh(self.inner.s6_addr[2]),
         ntoh(self.inner.s6_addr[3]),
         ntoh(self.inner.s6_addr[4]),
         ntoh(self.inner.s6_addr[5]),
         ntoh(self.inner.s6_addr[6]),
         ntoh(self.inner.s6_addr[7])]
    }
}

impl AsInner<libc::in6_addr> for Ipv6Addr {
    fn as_inner(&self) -> &libc::in6_addr {
        &self.inner
    }
}

impl IntoInner<libc::in6_addr> for Ipv6Addr {
    fn into_inner(self) -> libc::in6_addr {
        self.inner
    }
}

impl FromInner<libc::in6_addr> for Ipv6Addr {
    fn from_inner(inner: libc::in6_addr) -> Self {
        Ipv6Addr { inner: inner }
    }
}

impl Clone for Ipv6Addr {
    fn clone(&self) -> Ipv6Addr { *self }
}

impl PartialEq for Ipv6Addr {
    fn eq(&self, other: &Ipv6Addr) -> bool {
        self.inner.s6_addr == other.inner.s6_addr
    }
}

impl Eq for Ipv6Addr {}

impl hash::Hash for Ipv6Addr {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        self.inner.s6_addr.hash(s)
    }
}

impl PartialOrd for Ipv6Addr {
    fn partial_cmp(&self, other: &Ipv6Addr) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ipv6Addr {
    fn cmp(&self, other: &Ipv6Addr) -> Ordering {
        self.inner.s6_addr.cmp(&other.inner.s6_addr)
    }
}
