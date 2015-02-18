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

use cmp::Ordering;
use hash;
use fmt;
use libc;
use sys_common::{AsInner, FromInner};
use net::{hton, ntoh};

/// Representation of an IPv4 address.
#[derive(Copy)]
pub struct Ipv4Addr {
    inner: libc::in_addr,
}

/// Representation of an IPv6 address.
#[derive(Copy)]
pub struct Ipv6Addr {
    inner: libc::in6_addr,
}

#[allow(missing_docs)]
#[derive(Copy, PartialEq, Eq, Clone, Hash, Debug)]
pub enum Ipv6MulticastScope {
    InterfaceLocal,
    LinkLocal,
    RealmLocal,
    AdminLocal,
    SiteLocal,
    OrganizationLocal,
    Global
}

/// Enumeration of possible IP addresses
#[derive(Copy, PartialEq, Eq, Clone, Hash, Debug)]
pub enum IpAddr {
    /// An IPv4 address.
    V4(Ipv4Addr),
    /// An IPv6 address.
    V6(Ipv6Addr)
}

impl IpAddr {
    /// Create a new IpAddr that contains an IPv4 address.
    ///
    /// The result will represent the IP address a.b.c.d
    pub fn new_v4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(a, b, c, d))
    }

    /// Create a new IpAddr that contains an IPv6 address.
    ///
    /// The result will represent the IP address a:b:c:d:e:f
    pub fn new_v6(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16,
                  h: u16) -> IpAddr {
        IpAddr::V6(Ipv6Addr::new(a, b, c, d, e, f, g, h))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for IpAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            IpAddr::V4(v4) => v4.fmt(f),
            IpAddr::V6(v6) => v6.fmt(f)
        }
    }
}

impl Ipv4Addr {
    /// Create a new IPv4 address from four eight-bit octets.
    ///
    /// The result will represent the IP address a.b.c.d
    pub fn new(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr {
        Ipv4Addr {
            inner: libc::in_addr {
                s_addr: hton(((a as u32) << 24) |
                             ((b as u32) << 16) |
                             ((c as u32) <<  8) |
                              (d as u32)),
            }
        }
    }

    /// Returns the four eight-bit integers that make up this address
    pub fn octets(&self) -> [u8; 4] {
        let bits = ntoh(self.inner.s_addr);
        [(bits >> 24) as u8, (bits >> 16) as u8, (bits >> 8) as u8, bits as u8]
    }

    /// Returns true for the special 'unspecified' address 0.0.0.0
    pub fn is_unspecified(&self) -> bool {
        self.inner.s_addr == 0
    }

    /// Returns true if this is a loopback address (127.0.0.0/8)
    pub fn is_loopback(&self) -> bool {
        self.octets()[0] == 127
    }

    /// Returns true if this is a private address.
    ///
    /// The private address ranges are defined in RFC1918 and include:
    ///
    ///  - 10.0.0.0/8
    ///  - 172.16.0.0/12
    ///  - 192.168.0.0/16
    pub fn is_private(&self) -> bool {
        match (self.octets()[0], self.octets()[1]) {
            (10, _) => true,
            (172, b) if b >= 16 && b <= 31 => true,
            (192, 168) => true,
            _ => false
        }
    }

    /// Returns true if the address is link-local (169.254.0.0/16)
    pub fn is_link_local(&self) -> bool {
        self.octets()[0] == 169 && self.octets()[1] == 254
    }

    /// Returns true if the address appears to be globally routable.
    ///
    /// Non-globally-routable networks include the private networks (10.0.0.0/8,
    /// 172.16.0.0/12 and 192.168.0.0/16), the loopback network (127.0.0.0/8),
    /// and the link-local network (169.254.0.0/16).
    pub fn is_global(&self) -> bool {
        !self.is_private() && !self.is_loopback() && !self.is_link_local()
    }

    /// Returns true if this is a multicast address.
    ///
    /// Multicast addresses have a most significant octet between 224 and 239.
    pub fn is_multicast(&self) -> bool {
        self.octets()[0] >= 224 && self.octets()[0] <= 239
    }

    /// Convert this address to an IPv4-compatible IPv6 address
    ///
    /// a.b.c.d becomes ::a.b.c.d
    pub fn to_ipv6_compatible(&self) -> Ipv6Addr {
        Ipv6Addr::new(0, 0, 0, 0, 0, 0,
                      ((self.octets()[0] as u16) << 8) | self.octets()[1] as u16,
                      ((self.octets()[2] as u16) << 8) | self.octets()[3] as u16)
    }

    /// Convert this address to an IPv4-mapped IPv6 address
    ///
    /// a.b.c.d becomes ::ffff:a.b.c.d
    pub fn to_ipv6_mapped(&self) -> Ipv6Addr {
        Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff,
                      ((self.octets()[0] as u16) << 8) | self.octets()[1] as u16,
                      ((self.octets()[2] as u16) << 8) | self.octets()[3] as u16)
    }

}

impl fmt::Display for Ipv4Addr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let octets = self.octets();
        write!(fmt, "{}.{}.{}.{}", octets[0], octets[1], octets[2], octets[3])
    }
}

impl fmt::Debug for Ipv4Addr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
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

#[cfg(stage0)]
impl<S: hash::Hasher + hash::Writer> hash::Hash<S> for Ipv4Addr {
    fn hash(&self, s: &mut S) {
        self.inner.s_addr.hash(s)
    }
}
#[cfg(not(stage0))]
#[stable(feature = "rust1", since = "1.0.0")]
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

impl AsInner<libc::in_addr> for Ipv4Addr {
    fn as_inner(&self) -> &libc::in_addr { &self.inner }
}
impl FromInner<libc::in_addr> for Ipv4Addr {
    fn from_inner(addr: libc::in_addr) -> Ipv4Addr {
        Ipv4Addr { inner: addr }
    }
}

impl Ipv6Addr {
    /// Create a new IPv6 address from eight 16-bit segments.
    ///
    /// The result will represent the IP address a:b:c:d:e:f
    pub fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16,
               h: u16) -> Ipv6Addr {
        Ipv6Addr {
            inner: libc::in6_addr {
                s6_addr: [hton(a), hton(b), hton(c), hton(d),
                          hton(e), hton(f), hton(g), hton(h)]
            }
        }
    }

    /// Return the eight 16-bit segments that make up this address
    pub fn segments(&self) -> [u16; 8] {
        [ntoh(self.inner.s6_addr[0]),
         ntoh(self.inner.s6_addr[1]),
         ntoh(self.inner.s6_addr[2]),
         ntoh(self.inner.s6_addr[3]),
         ntoh(self.inner.s6_addr[4]),
         ntoh(self.inner.s6_addr[5]),
         ntoh(self.inner.s6_addr[6]),
         ntoh(self.inner.s6_addr[7])]
    }

    /// Returns true for the special 'unspecified' address ::
    pub fn is_unspecified(&self) -> bool {
        self.segments() == [0, 0, 0, 0, 0, 0, 0, 0]
    }

    /// Returns true if this is a loopback address (::1)
    pub fn is_loopback(&self) -> bool {
        self.segments() == [0, 0, 0, 0, 0, 0, 0, 1]
    }

    /// Returns true if the address appears to be globally routable.
    ///
    /// Non-globally-routable networks include the loopback address; the
    /// link-local, site-local, and unique local unicast addresses; and the
    /// interface-, link-, realm-, admin- and site-local multicast addresses.
    pub fn is_global(&self) -> bool {
        match self.multicast_scope() {
            Some(Ipv6MulticastScope::Global) => true,
            None => self.is_unicast_global(),
            _ => false
        }
    }

    /// Returns true if this is a unique local address (IPv6)
    ///
    /// Unique local addresses are defined in RFC4193 and have the form fc00::/7
    pub fn is_unique_local(&self) -> bool {
        (self.segments()[0] & 0xfe00) == 0xfc00
    }

    /// Returns true if the address is unicast and link-local (fe80::/10)
    pub fn is_unicast_link_local(&self) -> bool {
        (self.segments()[0] & 0xffc0) == 0xfe80
    }

    /// Returns true if this is a deprecated unicast site-local address (IPv6
    /// fec0::/10)
    pub fn is_unicast_site_local(&self) -> bool {
        (self.segments()[0] & 0xffc0) == 0xfec0
    }

    /// Returns true if the address is a globally routable unicast address
    ///
    /// Non-globally-routable unicast addresses include the loopback address,
    /// the link-local addresses, the deprecated site-local addresses and the
    /// unique local addresses.
    pub fn is_unicast_global(&self) -> bool {
        !self.is_multicast()
            && !self.is_loopback() && !self.is_unicast_link_local()
            && !self.is_unicast_site_local() && !self.is_unique_local()
    }

    /// Returns the address's multicast scope if the address is multicast.
    pub fn multicast_scope(&self) -> Option<Ipv6MulticastScope> {
        if self.is_multicast() {
            match self.segments()[0] & 0x000f {
                1 => Some(Ipv6MulticastScope::InterfaceLocal),
                2 => Some(Ipv6MulticastScope::LinkLocal),
                3 => Some(Ipv6MulticastScope::RealmLocal),
                4 => Some(Ipv6MulticastScope::AdminLocal),
                5 => Some(Ipv6MulticastScope::SiteLocal),
                8 => Some(Ipv6MulticastScope::OrganizationLocal),
                14 => Some(Ipv6MulticastScope::Global),
                _ => None
            }
        } else {
            None
        }
    }

    /// Returns true if this is a multicast address.
    ///
    /// Multicast addresses have the form ff00::/8.
    pub fn is_multicast(&self) -> bool {
        (self.segments()[0] & 0xff00) == 0xff00
    }

    /// Convert this address to an IPv4 address. Returns None if this address is
    /// neither IPv4-compatible or IPv4-mapped.
    ///
    /// ::a.b.c.d and ::ffff:a.b.c.d become a.b.c.d
    pub fn to_ipv4(&self) -> Option<Ipv4Addr> {
        match self.segments() {
            [0, 0, 0, 0, 0, f, g, h] if f == 0 || f == 0xffff => {
                Some(Ipv4Addr::new((g >> 8) as u8, g as u8,
                                   (h >> 8) as u8, h as u8))
            },
            _ => None
        }
    }
}

impl fmt::Display for Ipv6Addr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self.segments() {
            // We need special cases for :: and ::1, otherwise they're formatted
            // as ::0.0.0.[01]
            [0, 0, 0, 0, 0, 0, 0, 0] => write!(fmt, "::"),
            [0, 0, 0, 0, 0, 0, 0, 1] => write!(fmt, "::1"),
            // Ipv4 Compatible address
            [0, 0, 0, 0, 0, 0, g, h] => {
                write!(fmt, "::{}.{}.{}.{}", (g >> 8) as u8, g as u8,
                       (h >> 8) as u8, h as u8)
            }
            // Ipv4-Mapped address
            [0, 0, 0, 0, 0, 0xffff, g, h] => {
                write!(fmt, "::ffff:{}.{}.{}.{}", (g >> 8) as u8, g as u8,
                       (h >> 8) as u8, h as u8)
            },
            _ => {
                fn find_zero_slice(segments: &[u16; 8]) -> (usize, usize) {
                    let mut longest_span_len = 0;
                    let mut longest_span_at = 0;
                    let mut cur_span_len = 0;
                    let mut cur_span_at = 0;

                    for i in range(0, 8) {
                        if segments[i] == 0 {
                            if cur_span_len == 0 {
                                cur_span_at = i;
                            }

                            cur_span_len += 1;

                            if cur_span_len > longest_span_len {
                                longest_span_len = cur_span_len;
                                longest_span_at = cur_span_at;
                            }
                        } else {
                            cur_span_len = 0;
                            cur_span_at = 0;
                        }
                    }

                    (longest_span_at, longest_span_len)
                }

                let (zeros_at, zeros_len) = find_zero_slice(&self.segments());

                if zeros_len > 1 {
                    fn fmt_subslice(segments: &[u16]) -> String {
                        segments
                            .iter()
                            .map(|&seg| format!("{:x}", seg))
                            .collect::<Vec<String>>()
                            .as_slice()
                            .connect(":")
                    }

                    write!(fmt, "{}::{}",
                           fmt_subslice(&self.segments()[..zeros_at]),
                           fmt_subslice(&self.segments()[zeros_at + zeros_len..]))
                } else {
                    let &[a, b, c, d, e, f, g, h] = &self.segments();
                    write!(fmt, "{:x}:{:x}:{:x}:{:x}:{:x}:{:x}:{:x}:{:x}",
                           a, b, c, d, e, f, g, h)
                }
            }
        }
    }
}

impl fmt::Debug for Ipv6Addr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
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

#[cfg(stage0)]
impl<S: hash::Hasher + hash::Writer> hash::Hash<S> for Ipv6Addr {
    fn hash(&self, s: &mut S) {
        self.inner.s6_addr.hash(s)
    }
}
#[cfg(not(stage0))]
#[stable(feature = "rust1", since = "1.0.0")]
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

impl AsInner<libc::in6_addr> for Ipv6Addr {
    fn as_inner(&self) -> &libc::in6_addr { &self.inner }
}
impl FromInner<libc::in6_addr> for Ipv6Addr {
    fn from_inner(addr: libc::in6_addr) -> Ipv6Addr {
        Ipv6Addr { inner: addr }
    }
}
