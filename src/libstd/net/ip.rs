// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "ip", reason = "extra functionality has not been \
                                      scrutinized to the level that it should \
                                      be stable",
            issue = "27709")]

use prelude::v1::*;

use cmp::Ordering;
use fmt;
use hash;
use mem;
use net::{hton, ntoh};
use sys::net::netc as c;
use sys_common::{AsInner, FromInner};

/// An IP address, either an IPv4 or IPv6 address.
#[unstable(feature = "ip_addr", reason = "recent addition", issue = "27801")]
#[rustc_deprecated(reason = "too small a type to pull its weight",
                   since = "1.6.0")]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash, PartialOrd, Ord)]
#[allow(deprecated)]
pub enum IpAddr {
    /// Representation of an IPv4 address.
    V4(Ipv4Addr),
    /// Representation of an IPv6 address.
    V6(Ipv6Addr),
}

/// Representation of an IPv4 address.
#[derive(Copy)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Ipv4Addr {
    inner: c::in_addr,
}

/// Representation of an IPv6 address.
#[derive(Copy)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Ipv6Addr {
    inner: c::in6_addr,
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

impl Ipv4Addr {
    /// Creates a new IPv4 address from four eight-bit octets.
    ///
    /// The result will represent the IP address `a`.`b`.`c`.`d`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr {
        Ipv4Addr {
            inner: c::in_addr {
                s_addr: hton(((a as u32) << 24) |
                             ((b as u32) << 16) |
                             ((c as u32) <<  8) |
                              (d as u32)),
            }
        }
    }

    /// Returns the four eight-bit integers that make up this address.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn octets(&self) -> [u8; 4] {
        let bits = ntoh(self.inner.s_addr);
        [(bits >> 24) as u8, (bits >> 16) as u8, (bits >> 8) as u8, bits as u8]
    }

    /// Returns true for the special 'unspecified' address 0.0.0.0.
    pub fn is_unspecified(&self) -> bool {
        self.inner.s_addr == 0
    }

    /// Returns true if this is a loopback address (127.0.0.0/8).
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

    /// Returns true if the address is link-local (169.254.0.0/16).
    pub fn is_link_local(&self) -> bool {
        self.octets()[0] == 169 && self.octets()[1] == 254
    }

    /// Returns true if the address appears to be globally routable.
    ///
    /// The following return false:
    ///
    /// - private address (10.0.0.0/8, 172.16.0.0/12 and 192.168.0.0/16)
    /// - the loopback address (127.0.0.0/8)
    /// - the link-local address (169.254.0.0/16)
    /// - the broadcast address (255.255.255.255/32)
    /// - test addresses used for documentation (192.0.2.0/24, 198.51.100.0/24 and 203.0.113.0/24)
    pub fn is_global(&self) -> bool {
        !self.is_private() && !self.is_loopback() && !self.is_link_local() &&
        !self.is_broadcast() && !self.is_documentation()
    }

    /// Returns true if this is a multicast address.
    ///
    /// Multicast addresses have a most significant octet between 224 and 239.
    pub fn is_multicast(&self) -> bool {
        self.octets()[0] >= 224 && self.octets()[0] <= 239
    }

    /// Returns true if this is a broadcast address.
    ///
    /// A broadcast address has all octets set to 255 as defined in RFC 919.
    pub fn is_broadcast(&self) -> bool {
        self.octets()[0] == 255 && self.octets()[1] == 255 &&
        self.octets()[2] == 255 && self.octets()[3] == 255
    }

    /// Returns true if this address is in a range designated for documentation.
    ///
    /// This is defined in RFC 5737:
    ///
    /// - 192.0.2.0/24 (TEST-NET-1)
    /// - 198.51.100.0/24 (TEST-NET-2)
    /// - 203.0.113.0/24 (TEST-NET-3)
    pub fn is_documentation(&self) -> bool {
        match(self.octets()[0], self.octets()[1], self.octets()[2], self.octets()[3]) {
            (192, 0, 2, _) => true,
            (198, 51, 100, _) => true,
            (203, 0, 113, _) => true,
            _ => false
        }
    }

    /// Converts this address to an IPv4-compatible IPv6 address.
    ///
    /// a.b.c.d becomes ::a.b.c.d
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_ipv6_compatible(&self) -> Ipv6Addr {
        Ipv6Addr::new(0, 0, 0, 0, 0, 0,
                      ((self.octets()[0] as u16) << 8) | self.octets()[1] as u16,
                      ((self.octets()[2] as u16) << 8) | self.octets()[3] as u16)
    }

    /// Converts this address to an IPv4-mapped IPv6 address.
    ///
    /// a.b.c.d becomes ::ffff:a.b.c.d
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_ipv6_mapped(&self) -> Ipv6Addr {
        Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff,
                      ((self.octets()[0] as u16) << 8) | self.octets()[1] as u16,
                      ((self.octets()[2] as u16) << 8) | self.octets()[3] as u16)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
impl fmt::Display for IpAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            IpAddr::V4(ref a) => a.fmt(fmt),
            IpAddr::V6(ref a) => a.fmt(fmt),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for Ipv4Addr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let octets = self.octets();
        write!(fmt, "{}.{}.{}.{}", octets[0], octets[1], octets[2], octets[3])
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Ipv4Addr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Clone for Ipv4Addr {
    fn clone(&self) -> Ipv4Addr { *self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for Ipv4Addr {
    fn eq(&self, other: &Ipv4Addr) -> bool {
        self.inner.s_addr == other.inner.s_addr
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for Ipv4Addr {}

#[stable(feature = "rust1", since = "1.0.0")]
impl hash::Hash for Ipv4Addr {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        self.inner.s_addr.hash(s)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for Ipv4Addr {
    fn partial_cmp(&self, other: &Ipv4Addr) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for Ipv4Addr {
    fn cmp(&self, other: &Ipv4Addr) -> Ordering {
        self.octets().cmp(&other.octets())
    }
}

impl AsInner<c::in_addr> for Ipv4Addr {
    fn as_inner(&self) -> &c::in_addr { &self.inner }
}
impl FromInner<c::in_addr> for Ipv4Addr {
    fn from_inner(addr: c::in_addr) -> Ipv4Addr {
        Ipv4Addr { inner: addr }
    }
}

#[stable(feature = "ip_u32", since = "1.1.0")]
impl From<Ipv4Addr> for u32 {
    fn from(ip: Ipv4Addr) -> u32 {
        let ip = ip.octets();
        ((ip[0] as u32) << 24) + ((ip[1] as u32) << 16) + ((ip[2] as u32) << 8) + (ip[3] as u32)
    }
}

#[stable(feature = "ip_u32", since = "1.1.0")]
impl From<u32> for Ipv4Addr {
    fn from(ip: u32) -> Ipv4Addr {
        Ipv4Addr::new((ip >> 24) as u8, (ip >> 16) as u8, (ip >> 8) as u8, ip as u8)
    }
}

impl Ipv6Addr {
    /// Creates a new IPv6 address from eight 16-bit segments.
    ///
    /// The result will represent the IP address a:b:c:d:e:f:g:h.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16,
               h: u16) -> Ipv6Addr {
        let mut addr: c::in6_addr = unsafe { mem::zeroed() };
        addr.s6_addr = [(a >> 8) as u8, a as u8,
                        (b >> 8) as u8, b as u8,
                        (c >> 8) as u8, c as u8,
                        (d >> 8) as u8, d as u8,
                        (e >> 8) as u8, e as u8,
                        (f >> 8) as u8, f as u8,
                        (g >> 8) as u8, g as u8,
                        (h >> 8) as u8, h as u8];
        Ipv6Addr { inner: addr }
    }

    /// Returns the eight 16-bit segments that make up this address.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn segments(&self) -> [u16; 8] {
        let arr = &self.inner.s6_addr;
        [
            (arr[0] as u16) << 8 | (arr[1] as u16),
            (arr[2] as u16) << 8 | (arr[3] as u16),
            (arr[4] as u16) << 8 | (arr[5] as u16),
            (arr[6] as u16) << 8 | (arr[7] as u16),
            (arr[8] as u16) << 8 | (arr[9] as u16),
            (arr[10] as u16) << 8 | (arr[11] as u16),
            (arr[12] as u16) << 8 | (arr[13] as u16),
            (arr[14] as u16) << 8 | (arr[15] as u16),
        ]
    }

    /// Returns true for the special 'unspecified' address ::.
    pub fn is_unspecified(&self) -> bool {
        self.segments() == [0, 0, 0, 0, 0, 0, 0, 0]
    }

    /// Returns true if this is a loopback address (::1).
    pub fn is_loopback(&self) -> bool {
        self.segments() == [0, 0, 0, 0, 0, 0, 0, 1]
    }

    /// Returns true if the address appears to be globally routable.
    ///
    /// The following return false:
    ///
    /// - the loopback address
    /// - link-local, site-local, and unique local unicast addresses
    /// - interface-, link-, realm-, admin- and site-local multicast addresses
    pub fn is_global(&self) -> bool {
        match self.multicast_scope() {
            Some(Ipv6MulticastScope::Global) => true,
            None => self.is_unicast_global(),
            _ => false
        }
    }

    /// Returns true if this is a unique local address (IPv6).
    ///
    /// Unique local addresses are defined in RFC4193 and have the form fc00::/7.
    pub fn is_unique_local(&self) -> bool {
        (self.segments()[0] & 0xfe00) == 0xfc00
    }

    /// Returns true if the address is unicast and link-local (fe80::/10).
    pub fn is_unicast_link_local(&self) -> bool {
        (self.segments()[0] & 0xffc0) == 0xfe80
    }

    /// Returns true if this is a deprecated unicast site-local address (IPv6
    /// fec0::/10).
    pub fn is_unicast_site_local(&self) -> bool {
        (self.segments()[0] & 0xffc0) == 0xfec0
    }

    /// Returns true if the address is a globally routable unicast address.
    ///
    /// The following return false:
    ///
    /// - the loopback address
    /// - the link-local addresses
    /// - the (deprecated) site-local addresses
    /// - unique local addresses
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

    /// Converts this address to an IPv4 address. Returns None if this address is
    /// neither IPv4-compatible or IPv4-mapped.
    ///
    /// ::a.b.c.d and ::ffff:a.b.c.d become a.b.c.d
    #[stable(feature = "rust1", since = "1.0.0")]
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

#[stable(feature = "rust1", since = "1.0.0")]
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

                    for i in 0..8 {
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
                            .join(":")
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

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Ipv6Addr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Clone for Ipv6Addr {
    fn clone(&self) -> Ipv6Addr { *self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for Ipv6Addr {
    fn eq(&self, other: &Ipv6Addr) -> bool {
        self.inner.s6_addr == other.inner.s6_addr
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for Ipv6Addr {}

#[stable(feature = "rust1", since = "1.0.0")]
impl hash::Hash for Ipv6Addr {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        self.inner.s6_addr.hash(s)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for Ipv6Addr {
    fn partial_cmp(&self, other: &Ipv6Addr) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for Ipv6Addr {
    fn cmp(&self, other: &Ipv6Addr) -> Ordering {
        self.segments().cmp(&other.segments())
    }
}

impl AsInner<c::in6_addr> for Ipv6Addr {
    fn as_inner(&self) -> &c::in6_addr { &self.inner }
}
impl FromInner<c::in6_addr> for Ipv6Addr {
    fn from_inner(addr: c::in6_addr) -> Ipv6Addr {
        Ipv6Addr { inner: addr }
    }
}

// Tests for this module
#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use io;
    use net::*;
    use net::Ipv6MulticastScope::*;
    use net::test::{tsa, sa6, sa4};

    #[test]
    fn test_from_str_ipv4() {
        assert_eq!(Ok(Ipv4Addr::new(127, 0, 0, 1)), "127.0.0.1".parse());
        assert_eq!(Ok(Ipv4Addr::new(255, 255, 255, 255)), "255.255.255.255".parse());
        assert_eq!(Ok(Ipv4Addr::new(0, 0, 0, 0)), "0.0.0.0".parse());

        // out of range
        let none: Option<Ipv4Addr> = "256.0.0.1".parse().ok();
        assert_eq!(None, none);
        // too short
        let none: Option<Ipv4Addr> = "255.0.0".parse().ok();
        assert_eq!(None, none);
        // too long
        let none: Option<Ipv4Addr> = "255.0.0.1.2".parse().ok();
        assert_eq!(None, none);
        // no number between dots
        let none: Option<Ipv4Addr> = "255.0..1".parse().ok();
        assert_eq!(None, none);
    }

    #[test]
    fn test_from_str_ipv6() {
        assert_eq!(Ok(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0)), "0:0:0:0:0:0:0:0".parse());
        assert_eq!(Ok(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1)), "0:0:0:0:0:0:0:1".parse());

        assert_eq!(Ok(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1)), "::1".parse());
        assert_eq!(Ok(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0)), "::".parse());

        assert_eq!(Ok(Ipv6Addr::new(0x2a02, 0x6b8, 0, 0, 0, 0, 0x11, 0x11)),
                "2a02:6b8::11:11".parse());

        // too long group
        let none: Option<Ipv6Addr> = "::00000".parse().ok();
        assert_eq!(None, none);
        // too short
        let none: Option<Ipv6Addr> = "1:2:3:4:5:6:7".parse().ok();
        assert_eq!(None, none);
        // too long
        let none: Option<Ipv6Addr> = "1:2:3:4:5:6:7:8:9".parse().ok();
        assert_eq!(None, none);
        // triple colon
        let none: Option<Ipv6Addr> = "1:2:::6:7:8".parse().ok();
        assert_eq!(None, none);
        // two double colons
        let none: Option<Ipv6Addr> = "1:2::6::8".parse().ok();
        assert_eq!(None, none);
    }

    #[test]
    fn test_from_str_ipv4_in_ipv6() {
        assert_eq!(Ok(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 49152, 545)),
                "::192.0.2.33".parse());
        assert_eq!(Ok(Ipv6Addr::new(0, 0, 0, 0, 0, 0xFFFF, 49152, 545)),
                "::FFFF:192.0.2.33".parse());
        assert_eq!(Ok(Ipv6Addr::new(0x64, 0xff9b, 0, 0, 0, 0, 49152, 545)),
                "64:ff9b::192.0.2.33".parse());
        assert_eq!(Ok(Ipv6Addr::new(0x2001, 0xdb8, 0x122, 0xc000, 0x2, 0x2100, 49152, 545)),
                "2001:db8:122:c000:2:2100:192.0.2.33".parse());

        // colon after v4
        let none: Option<Ipv4Addr> = "::127.0.0.1:".parse().ok();
        assert_eq!(None, none);
        // not enough groups
        let none: Option<Ipv6Addr> = "1.2.3.4.5:127.0.0.1".parse().ok();
        assert_eq!(None, none);
        // too many groups
        let none: Option<Ipv6Addr> = "1.2.3.4.5:6:7:127.0.0.1".parse().ok();
        assert_eq!(None, none);
    }

    #[test]
    fn test_from_str_socket_addr() {
        assert_eq!(Ok(sa4(Ipv4Addr::new(77, 88, 21, 11), 80)),
                   "77.88.21.11:80".parse());
        assert_eq!(Ok(SocketAddrV4::new(Ipv4Addr::new(77, 88, 21, 11), 80)),
                   "77.88.21.11:80".parse());
        assert_eq!(Ok(sa6(Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 53)),
                   "[2a02:6b8:0:1::1]:53".parse());
        assert_eq!(Ok(SocketAddrV6::new(Ipv6Addr::new(0x2a02, 0x6b8, 0, 1,
                                                      0, 0, 0, 1), 53, 0, 0)),
                   "[2a02:6b8:0:1::1]:53".parse());
        assert_eq!(Ok(sa6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0x7F00, 1), 22)),
                   "[::127.0.0.1]:22".parse());
        assert_eq!(Ok(SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0,
                                                      0x7F00, 1), 22, 0, 0)),
                   "[::127.0.0.1]:22".parse());

        // without port
        let none: Option<SocketAddr> = "127.0.0.1".parse().ok();
        assert_eq!(None, none);
        // without port
        let none: Option<SocketAddr> = "127.0.0.1:".parse().ok();
        assert_eq!(None, none);
        // wrong brackets around v4
        let none: Option<SocketAddr> = "[127.0.0.1]:22".parse().ok();
        assert_eq!(None, none);
        // port out of range
        let none: Option<SocketAddr> = "127.0.0.1:123456".parse().ok();
        assert_eq!(None, none);
    }

    #[test]
    fn ipv6_addr_to_string() {
        // ipv4-mapped address
        let a1 = Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc000, 0x280);
        assert_eq!(a1.to_string(), "::ffff:192.0.2.128");

        // ipv4-compatible address
        let a1 = Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0xc000, 0x280);
        assert_eq!(a1.to_string(), "::192.0.2.128");

        // v6 address with no zero segments
        assert_eq!(Ipv6Addr::new(8, 9, 10, 11, 12, 13, 14, 15).to_string(),
                   "8:9:a:b:c:d:e:f");

        // reduce a single run of zeros
        assert_eq!("ae::ffff:102:304",
                   Ipv6Addr::new(0xae, 0, 0, 0, 0, 0xffff, 0x0102, 0x0304).to_string());

        // don't reduce just a single zero segment
        assert_eq!("1:2:3:4:5:6:0:8",
                   Ipv6Addr::new(1, 2, 3, 4, 5, 6, 0, 8).to_string());

        // 'any' address
        assert_eq!("::", Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0).to_string());

        // loopback address
        assert_eq!("::1", Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1).to_string());

        // ends in zeros
        assert_eq!("1::", Ipv6Addr::new(1, 0, 0, 0, 0, 0, 0, 0).to_string());

        // two runs of zeros, second one is longer
        assert_eq!("1:0:0:4::8", Ipv6Addr::new(1, 0, 0, 4, 0, 0, 0, 8).to_string());

        // two runs of zeros, equal length
        assert_eq!("1::4:5:0:0:8", Ipv6Addr::new(1, 0, 0, 4, 5, 0, 0, 8).to_string());
    }

    #[test]
    fn ipv4_to_ipv6() {
        assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x1234, 0x5678),
                   Ipv4Addr::new(0x12, 0x34, 0x56, 0x78).to_ipv6_mapped());
        assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0x1234, 0x5678),
                   Ipv4Addr::new(0x12, 0x34, 0x56, 0x78).to_ipv6_compatible());
    }

    #[test]
    fn ipv6_to_ipv4() {
        assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x1234, 0x5678).to_ipv4(),
                   Some(Ipv4Addr::new(0x12, 0x34, 0x56, 0x78)));
        assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0x1234, 0x5678).to_ipv4(),
                   Some(Ipv4Addr::new(0x12, 0x34, 0x56, 0x78)));
        assert_eq!(Ipv6Addr::new(0, 0, 1, 0, 0, 0, 0x1234, 0x5678).to_ipv4(),
                   None);
    }

    #[test]
    fn ipv4_properties() {
        fn check(octets: &[u8; 4], unspec: bool, loopback: bool,
                 private: bool, link_local: bool, global: bool,
                 multicast: bool, broadcast: bool, documentation: bool) {
            let ip = Ipv4Addr::new(octets[0], octets[1], octets[2], octets[3]);
            assert_eq!(octets, &ip.octets());

            assert_eq!(ip.is_unspecified(), unspec);
            assert_eq!(ip.is_loopback(), loopback);
            assert_eq!(ip.is_private(), private);
            assert_eq!(ip.is_link_local(), link_local);
            assert_eq!(ip.is_global(), global);
            assert_eq!(ip.is_multicast(), multicast);
            assert_eq!(ip.is_broadcast(), broadcast);
            assert_eq!(ip.is_documentation(), documentation);
        }

        //    address                unspec loopbk privt  linloc global multicast brdcast doc
        check(&[0, 0, 0, 0],         true,  false, false, false, true,  false,    false,  false);
        check(&[0, 0, 0, 1],         false, false, false, false, true,  false,    false,  false);
        check(&[1, 0, 0, 0],         false, false, false, false, true,  false,    false,  false);
        check(&[10, 9, 8, 7],        false, false, true,  false, false, false,    false,  false);
        check(&[127, 1, 2, 3],       false, true,  false, false, false, false,    false,  false);
        check(&[172, 31, 254, 253],  false, false, true,  false, false, false,    false,  false);
        check(&[169, 254, 253, 242], false, false, false, true,  false, false,    false,  false);
        check(&[192, 0, 2, 183],     false, false, false, false, false, false,    false,  true);
        check(&[192, 1, 2, 183],     false, false, false, false, true,  false,    false,  false);
        check(&[192, 168, 254, 253], false, false, true,  false, false, false,    false,  false);
        check(&[198, 51, 100, 0],    false, false, false, false, false, false,    false,  true);
        check(&[203, 0, 113, 0],     false, false, false, false, false, false,    false,  true);
        check(&[203, 2, 113, 0],     false, false, false, false, true,  false,    false,  false);
        check(&[224, 0, 0, 0],       false, false, false, false, true,  true,     false,  false);
        check(&[239, 255, 255, 255], false, false, false, false, true,  true,     false,  false);
        check(&[255, 255, 255, 255], false, false, false, false, false, false,    true,   false);
    }

    #[test]
    fn ipv6_properties() {
        fn check(str_addr: &str, unspec: bool, loopback: bool,
                 unique_local: bool, global: bool,
                 u_link_local: bool, u_site_local: bool, u_global: bool,
                 m_scope: Option<Ipv6MulticastScope>) {
            let ip: Ipv6Addr = str_addr.parse().unwrap();
            assert_eq!(str_addr, ip.to_string());

            assert_eq!(ip.is_unspecified(), unspec);
            assert_eq!(ip.is_loopback(), loopback);
            assert_eq!(ip.is_unique_local(), unique_local);
            assert_eq!(ip.is_global(), global);
            assert_eq!(ip.is_unicast_link_local(), u_link_local);
            assert_eq!(ip.is_unicast_site_local(), u_site_local);
            assert_eq!(ip.is_unicast_global(), u_global);
            assert_eq!(ip.multicast_scope(), m_scope);
            assert_eq!(ip.is_multicast(), m_scope.is_some());
        }

        //    unspec loopbk uniqlo global unill  unisl  uniglo mscope
        check("::",
              true,  false, false, true,  false, false, true,  None);
        check("::1",
              false, true,  false, false, false, false, false, None);
        check("::0.0.0.2",
              false, false, false, true,  false, false, true,  None);
        check("1::",
              false, false, false, true,  false, false, true,  None);
        check("fc00::",
              false, false, true,  false, false, false, false, None);
        check("fdff:ffff::",
              false, false, true,  false, false, false, false, None);
        check("fe80:ffff::",
              false, false, false, false, true,  false, false, None);
        check("febf:ffff::",
              false, false, false, false, true,  false, false, None);
        check("fec0::",
              false, false, false, false, false, true,  false, None);
        check("ff01::",
              false, false, false, false, false, false, false, Some(InterfaceLocal));
        check("ff02::",
              false, false, false, false, false, false, false, Some(LinkLocal));
        check("ff03::",
              false, false, false, false, false, false, false, Some(RealmLocal));
        check("ff04::",
              false, false, false, false, false, false, false, Some(AdminLocal));
        check("ff05::",
              false, false, false, false, false, false, false, Some(SiteLocal));
        check("ff08::",
              false, false, false, false, false, false, false, Some(OrganizationLocal));
        check("ff0e::",
              false, false, false, true,  false, false, false, Some(Global));
    }

    #[test]
    fn to_socket_addr_socketaddr() {
        let a = sa4(Ipv4Addr::new(77, 88, 21, 11), 12345);
        assert_eq!(Ok(vec![a]), tsa(a));
    }

    #[test]
    fn test_ipv4_to_int() {
        let a = Ipv4Addr::new(127, 0, 0, 1);
        assert_eq!(u32::from(a), 2130706433);
    }

    #[test]
    fn test_int_to_ipv4() {
        let a = Ipv4Addr::new(127, 0, 0, 1);
        assert_eq!(Ipv4Addr::from(2130706433), a);
    }

    #[test]
    fn ord() {
        assert!(Ipv4Addr::new(100, 64, 3, 3) < Ipv4Addr::new(192, 0, 2, 2));
        assert!("2001:db8:f00::1002".parse::<Ipv6Addr>().unwrap() <
                "2001:db8:f00::2001".parse::<Ipv6Addr>().unwrap());
    }
}
