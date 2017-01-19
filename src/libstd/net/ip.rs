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

use cmp::Ordering;
use fmt;
use hash;
use mem;
use net::{hton, ntoh};
use sys::net::netc as c;
use sys_common::{AsInner, FromInner};

/// An IP address, either an IPv4 or IPv6 address.
///
/// # Examples
///
/// Constructing an IPv4 address:
///
/// ```
/// use std::net::{IpAddr, Ipv4Addr};
///
/// IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
/// ```
///
/// Constructing an IPv6 address:
///
/// ```
/// use std::net::{IpAddr, Ipv6Addr};
///
/// IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1));
/// ```
#[stable(feature = "ip_addr", since = "1.7.0")]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash, PartialOrd, Ord)]
pub enum IpAddr {
    /// Representation of an IPv4 address.
    #[stable(feature = "ip_addr", since = "1.7.0")]
    V4(#[stable(feature = "ip_addr", since = "1.7.0")] Ipv4Addr),
    /// Representation of an IPv6 address.
    #[stable(feature = "ip_addr", since = "1.7.0")]
    V6(#[stable(feature = "ip_addr", since = "1.7.0")] Ipv6Addr),
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

impl IpAddr {
    /// Returns true for the special 'unspecified' address ([IPv4], [IPv6]).
    ///
    /// [IPv4]: ../../std/net/struct.Ipv4Addr.html#method.is_unspecified
    /// [IPv6]: ../../std/net/struct.Ipv6Addr.html#method.is_unspecified
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)).is_unspecified(), true);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0)).is_unspecified(), true);
    /// ```
    #[stable(feature = "ip_shared", since = "1.12.0")]
    pub fn is_unspecified(&self) -> bool {
        match *self {
            IpAddr::V4(ref a) => a.is_unspecified(),
            IpAddr::V6(ref a) => a.is_unspecified(),
        }
    }

    /// Returns true if this is a loopback address ([IPv4], [IPv6]).
    ///
    /// [IPv4]: ../../std/net/struct.Ipv4Addr.html#method.is_loopback
    /// [IPv6]: ../../std/net/struct.Ipv6Addr.html#method.is_loopback
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)).is_loopback(), true);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0x1)).is_loopback(), true);
    /// ```
    #[stable(feature = "ip_shared", since = "1.12.0")]
    pub fn is_loopback(&self) -> bool {
        match *self {
            IpAddr::V4(ref a) => a.is_loopback(),
            IpAddr::V6(ref a) => a.is_loopback(),
        }
    }

    /// Returns true if the address appears to be globally routable ([IPv4], [IPv6]).
    ///
    /// [IPv4]: ../../std/net/struct.Ipv4Addr.html#method.is_global
    /// [IPv6]: ../../std/net/struct.Ipv6Addr.html#method.is_global
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// fn main() {
    ///     assert_eq!(IpAddr::V4(Ipv4Addr::new(80, 9, 12, 3)).is_global(), true);
    ///     assert_eq!(IpAddr::V6(Ipv6Addr::new(0, 0, 0x1c9, 0, 0, 0xafc8, 0, 0x1)).is_global(),
    ///                true);
    /// }
    /// ```
    pub fn is_global(&self) -> bool {
        match *self {
            IpAddr::V4(ref a) => a.is_global(),
            IpAddr::V6(ref a) => a.is_global(),
        }
    }

    /// Returns true if this is a multicast address ([IPv4], [IPv6]).
    ///
    /// [IPv4]: ../../std/net/struct.Ipv4Addr.html#method.is_multicast
    /// [IPv6]: ../../std/net/struct.Ipv6Addr.html#method.is_multicast
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(224, 254, 0, 0)).is_multicast(), true);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0)).is_multicast(), true);
    /// ```
    #[stable(feature = "ip_shared", since = "1.12.0")]
    pub fn is_multicast(&self) -> bool {
        match *self {
            IpAddr::V4(ref a) => a.is_multicast(),
            IpAddr::V6(ref a) => a.is_multicast(),
        }
    }

    /// Returns true if this address is in a range designated for documentation ([IPv4], [IPv6]).
    ///
    /// [IPv4]: ../../std/net/struct.Ipv4Addr.html#method.is_documentation
    /// [IPv6]: ../../std/net/struct.Ipv6Addr.html#method.is_documentation
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// fn main() {
    ///     assert_eq!(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 6)).is_documentation(), true);
    ///     assert_eq!(IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0))
    ///                       .is_documentation(), true);
    /// }
    /// ```
    pub fn is_documentation(&self) -> bool {
        match *self {
            IpAddr::V4(ref a) => a.is_documentation(),
            IpAddr::V6(ref a) => a.is_documentation(),
        }
    }

    /// Returns true if this address is a valid IPv4 address, false if it's a valid IPv6 address.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ipaddr_checker)]
    ///
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// fn main() {
    ///     assert_eq!(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 6)).is_ipv4(), true);
    ///     assert_eq!(IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0)).is_ipv4(),
    ///                false);
    /// }
    /// ```
    #[unstable(feature = "ipaddr_checker", issue = "36949")]
    pub fn is_ipv4(&self) -> bool {
        match *self {
            IpAddr::V4(_) => true,
            IpAddr::V6(_) => false,
        }
    }

    /// Returns true if this address is a valid IPv6 address, false if it's a valid IPv4 address.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ipaddr_checker)]
    ///
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// fn main() {
    ///     assert_eq!(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 6)).is_ipv6(), false);
    ///     assert_eq!(IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0)).is_ipv6(),
    ///                true);
    /// }
    /// ```
    #[unstable(feature = "ipaddr_checker", issue = "36949")]
    pub fn is_ipv6(&self) -> bool {
        match *self {
            IpAddr::V4(_) => false,
            IpAddr::V6(_) => true,
        }
    }
}

impl Ipv4Addr {
    /// Creates a new IPv4 address from four eight-bit octets.
    ///
    /// The result will represent the IP address `a`.`b`.`c`.`d`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// let addr = Ipv4Addr::new(127, 0, 0, 1);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// let addr = Ipv4Addr::new(127, 0, 0, 1);
    /// assert_eq!(addr.octets(), [127, 0, 0, 1]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn octets(&self) -> [u8; 4] {
        let bits = ntoh(self.inner.s_addr);
        [(bits >> 24) as u8, (bits >> 16) as u8, (bits >> 8) as u8, bits as u8]
    }

    /// Returns true for the special 'unspecified' address (0.0.0.0).
    ///
    /// This property is defined in _UNIX Network Programming, Second Edition_,
    /// W. Richard Stevens, p. 891; see also [ip7].
    ///
    /// [ip7]: http://man7.org/linux/man-pages/man7/ip.7.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(0, 0, 0, 0).is_unspecified(), true);
    /// assert_eq!(Ipv4Addr::new(45, 22, 13, 197).is_unspecified(), false);
    /// ```
    #[stable(feature = "ip_shared", since = "1.12.0")]
    pub fn is_unspecified(&self) -> bool {
        self.inner.s_addr == 0
    }

    /// Returns true if this is a loopback address (127.0.0.0/8).
    ///
    /// This property is defined by [RFC 1122].
    ///
    /// [RFC 1122]: https://tools.ietf.org/html/rfc1122
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(127, 0, 0, 1).is_loopback(), true);
    /// assert_eq!(Ipv4Addr::new(45, 22, 13, 197).is_loopback(), false);
    /// ```
    #[stable(since = "1.7.0", feature = "ip_17")]
    pub fn is_loopback(&self) -> bool {
        self.octets()[0] == 127
    }

    /// Returns true if this is a private address.
    ///
    /// The private address ranges are defined in [RFC 1918] and include:
    ///
    ///  - 10.0.0.0/8
    ///  - 172.16.0.0/12
    ///  - 192.168.0.0/16
    ///
    /// [RFC 1918]: https://tools.ietf.org/html/rfc1918
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(10, 0, 0, 1).is_private(), true);
    /// assert_eq!(Ipv4Addr::new(10, 10, 10, 10).is_private(), true);
    /// assert_eq!(Ipv4Addr::new(172, 16, 10, 10).is_private(), true);
    /// assert_eq!(Ipv4Addr::new(172, 29, 45, 14).is_private(), true);
    /// assert_eq!(Ipv4Addr::new(172, 32, 0, 2).is_private(), false);
    /// assert_eq!(Ipv4Addr::new(192, 168, 0, 2).is_private(), true);
    /// assert_eq!(Ipv4Addr::new(192, 169, 0, 2).is_private(), false);
    /// ```
    #[stable(since = "1.7.0", feature = "ip_17")]
    pub fn is_private(&self) -> bool {
        match (self.octets()[0], self.octets()[1]) {
            (10, _) => true,
            (172, b) if b >= 16 && b <= 31 => true,
            (192, 168) => true,
            _ => false
        }
    }

    /// Returns true if the address is link-local (169.254.0.0/16).
    ///
    /// This property is defined by [RFC 3927].
    ///
    /// [RFC 3927]: https://tools.ietf.org/html/rfc3927
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(169, 254, 0, 0).is_link_local(), true);
    /// assert_eq!(Ipv4Addr::new(169, 254, 10, 65).is_link_local(), true);
    /// assert_eq!(Ipv4Addr::new(16, 89, 10, 65).is_link_local(), false);
    /// ```
    #[stable(since = "1.7.0", feature = "ip_17")]
    pub fn is_link_local(&self) -> bool {
        self.octets()[0] == 169 && self.octets()[1] == 254
    }

    /// Returns true if the address appears to be globally routable.
    /// See [iana-ipv4-special-registry][ipv4-sr].
    ///
    /// The following return false:
    ///
    /// - private address (10.0.0.0/8, 172.16.0.0/12 and 192.168.0.0/16)
    /// - the loopback address (127.0.0.0/8)
    /// - the link-local address (169.254.0.0/16)
    /// - the broadcast address (255.255.255.255/32)
    /// - test addresses used for documentation (192.0.2.0/24, 198.51.100.0/24 and 203.0.113.0/24)
    /// - the unspecified address (0.0.0.0)
    ///
    /// [ipv4-sr]: http://goo.gl/RaZ7lg
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv4Addr;
    ///
    /// fn main() {
    ///     assert_eq!(Ipv4Addr::new(10, 254, 0, 0).is_global(), false);
    ///     assert_eq!(Ipv4Addr::new(192, 168, 10, 65).is_global(), false);
    ///     assert_eq!(Ipv4Addr::new(172, 16, 10, 65).is_global(), false);
    ///     assert_eq!(Ipv4Addr::new(0, 0, 0, 0).is_global(), false);
    ///     assert_eq!(Ipv4Addr::new(80, 9, 12, 3).is_global(), true);
    /// }
    /// ```
    pub fn is_global(&self) -> bool {
        !self.is_private() && !self.is_loopback() && !self.is_link_local() &&
        !self.is_broadcast() && !self.is_documentation() && !self.is_unspecified()
    }

    /// Returns true if this is a multicast address (224.0.0.0/4).
    ///
    /// Multicast addresses have a most significant octet between 224 and 239,
    /// and is defined by [RFC 5771].
    ///
    /// [RFC 5771]: https://tools.ietf.org/html/rfc5771
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(224, 254, 0, 0).is_multicast(), true);
    /// assert_eq!(Ipv4Addr::new(236, 168, 10, 65).is_multicast(), true);
    /// assert_eq!(Ipv4Addr::new(172, 16, 10, 65).is_multicast(), false);
    /// ```
    #[stable(since = "1.7.0", feature = "ip_17")]
    pub fn is_multicast(&self) -> bool {
        self.octets()[0] >= 224 && self.octets()[0] <= 239
    }

    /// Returns true if this is a broadcast address (255.255.255.255).
    ///
    /// A broadcast address has all octets set to 255 as defined in [RFC 919].
    ///
    /// [RFC 919]: https://tools.ietf.org/html/rfc919
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(255, 255, 255, 255).is_broadcast(), true);
    /// assert_eq!(Ipv4Addr::new(236, 168, 10, 65).is_broadcast(), false);
    /// ```
    #[stable(since = "1.7.0", feature = "ip_17")]
    pub fn is_broadcast(&self) -> bool {
        self.octets()[0] == 255 && self.octets()[1] == 255 &&
        self.octets()[2] == 255 && self.octets()[3] == 255
    }

    /// Returns true if this address is in a range designated for documentation.
    ///
    /// This is defined in [RFC 5737]:
    ///
    /// - 192.0.2.0/24 (TEST-NET-1)
    /// - 198.51.100.0/24 (TEST-NET-2)
    /// - 203.0.113.0/24 (TEST-NET-3)
    ///
    /// [RFC 5737]: https://tools.ietf.org/html/rfc5737
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(192, 0, 2, 255).is_documentation(), true);
    /// assert_eq!(Ipv4Addr::new(198, 51, 100, 65).is_documentation(), true);
    /// assert_eq!(Ipv4Addr::new(203, 0, 113, 6).is_documentation(), true);
    /// assert_eq!(Ipv4Addr::new(193, 34, 17, 19).is_documentation(), false);
    /// ```
    #[stable(since = "1.7.0", feature = "ip_17")]
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
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(Ipv4Addr::new(192, 0, 2, 255).to_ipv6_compatible(),
    ///            Ipv6Addr::new(0, 0, 0, 0, 0, 0, 49152, 767));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_ipv6_compatible(&self) -> Ipv6Addr {
        Ipv6Addr::new(0, 0, 0, 0, 0, 0,
                      ((self.octets()[0] as u16) << 8) | self.octets()[1] as u16,
                      ((self.octets()[2] as u16) << 8) | self.octets()[3] as u16)
    }

    /// Converts this address to an IPv4-mapped IPv6 address.
    ///
    /// a.b.c.d becomes ::ffff:a.b.c.d
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(Ipv4Addr::new(192, 0, 2, 255).to_ipv6_mapped(),
    ///            Ipv6Addr::new(0, 0, 0, 0, 0, 65535, 49152, 767));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_ipv6_mapped(&self) -> Ipv6Addr {
        Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff,
                      ((self.octets()[0] as u16) << 8) | self.octets()[1] as u16,
                      ((self.octets()[2] as u16) << 8) | self.octets()[3] as u16)
    }
}

#[stable(feature = "ip_addr", since = "1.7.0")]
impl fmt::Display for IpAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            IpAddr::V4(ref a) => a.fmt(fmt),
            IpAddr::V6(ref a) => a.fmt(fmt),
        }
    }
}

#[stable(feature = "ip_from_ip", since = "1.16.0")]
impl From<Ipv4Addr> for IpAddr {
    fn from(ipv4: Ipv4Addr) -> IpAddr {
        IpAddr::V4(ipv4)
    }
}

#[stable(feature = "ip_from_ip", since = "1.16.0")]
impl From<Ipv6Addr> for IpAddr {
    fn from(ipv6: Ipv6Addr) -> IpAddr {
        IpAddr::V6(ipv6)
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

#[stable(feature = "ip_cmp", since = "1.15.0")]
impl PartialEq<Ipv4Addr> for IpAddr {
    fn eq(&self, other: &Ipv4Addr) -> bool {
        match *self {
            IpAddr::V4(ref v4) => v4 == other,
            IpAddr::V6(_) => false,
        }
    }
}

#[stable(feature = "ip_cmp", since = "1.15.0")]
impl PartialEq<IpAddr> for Ipv4Addr {
    fn eq(&self, other: &IpAddr) -> bool {
        match *other {
            IpAddr::V4(ref v4) => self == v4,
            IpAddr::V6(_) => false,
        }
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

#[stable(feature = "ip_cmp", since = "1.15.0")]
impl PartialOrd<Ipv4Addr> for IpAddr {
    fn partial_cmp(&self, other: &Ipv4Addr) -> Option<Ordering> {
        match *self {
            IpAddr::V4(ref v4) => v4.partial_cmp(other),
            IpAddr::V6(_) => Some(Ordering::Greater),
        }
    }
}

#[stable(feature = "ip_cmp", since = "1.15.0")]
impl PartialOrd<IpAddr> for Ipv4Addr {
    fn partial_cmp(&self, other: &IpAddr) -> Option<Ordering> {
        match *other {
            IpAddr::V4(ref v4) => self.partial_cmp(v4),
            IpAddr::V6(_) => Some(Ordering::Less),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for Ipv4Addr {
    fn cmp(&self, other: &Ipv4Addr) -> Ordering {
        ntoh(self.inner.s_addr).cmp(&ntoh(other.inner.s_addr))
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

#[stable(feature = "from_slice_v4", since = "1.9.0")]
impl From<[u8; 4]> for Ipv4Addr {
    fn from(octets: [u8; 4]) -> Ipv4Addr {
        Ipv4Addr::new(octets[0], octets[1], octets[2], octets[3])
    }
}

impl Ipv6Addr {
    /// Creates a new IPv6 address from eight 16-bit segments.
    ///
    /// The result will represent the IP address a:b:c:d:e:f:g:h.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// let addr = Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).segments(),
    ///            [0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff]);
    /// ```
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

    /// Returns true for the special 'unspecified' address (::).
    ///
    /// This property is defined in [RFC 4291].
    ///
    /// [RFC 4291]: https://tools.ietf.org/html/rfc4291
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_unspecified(), false);
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0).is_unspecified(), true);
    /// ```
    #[stable(since = "1.7.0", feature = "ip_17")]
    pub fn is_unspecified(&self) -> bool {
        self.segments() == [0, 0, 0, 0, 0, 0, 0, 0]
    }

    /// Returns true if this is a loopback address (::1).
    ///
    /// This property is defined in [RFC 4291].
    ///
    /// [RFC 4291]: https://tools.ietf.org/html/rfc4291
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_loopback(), false);
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0x1).is_loopback(), true);
    /// ```
    #[stable(since = "1.7.0", feature = "ip_17")]
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
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// fn main() {
    ///     assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_global(), true);
    ///     assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0x1).is_global(), false);
    ///     assert_eq!(Ipv6Addr::new(0, 0, 0x1c9, 0, 0, 0xafc8, 0, 0x1).is_global(), true);
    /// }
    /// ```
    pub fn is_global(&self) -> bool {
        match self.multicast_scope() {
            Some(Ipv6MulticastScope::Global) => true,
            None => self.is_unicast_global(),
            _ => false
        }
    }

    /// Returns true if this is a unique local address (fc00::/7).
    ///
    /// This property is defined in [RFC 4193].
    ///
    /// [RFC 4193]: https://tools.ietf.org/html/rfc4193
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// fn main() {
    ///     assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_unique_local(),
    ///                false);
    ///     assert_eq!(Ipv6Addr::new(0xfc02, 0, 0, 0, 0, 0, 0, 0).is_unique_local(), true);
    /// }
    /// ```
    pub fn is_unique_local(&self) -> bool {
        (self.segments()[0] & 0xfe00) == 0xfc00
    }

    /// Returns true if the address is unicast and link-local (fe80::/10).
    ///
    /// This property is defined in [RFC 4291].
    ///
    /// [RFC 4291]: https://tools.ietf.org/html/rfc4291
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// fn main() {
    ///     assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_unicast_link_local(),
    ///                false);
    ///     assert_eq!(Ipv6Addr::new(0xfe8a, 0, 0, 0, 0, 0, 0, 0).is_unicast_link_local(), true);
    /// }
    /// ```
    pub fn is_unicast_link_local(&self) -> bool {
        (self.segments()[0] & 0xffc0) == 0xfe80
    }

    /// Returns true if this is a deprecated unicast site-local address
    /// (fec0::/10).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// fn main() {
    ///     assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_unicast_site_local(),
    ///                false);
    ///     assert_eq!(Ipv6Addr::new(0xfec2, 0, 0, 0, 0, 0, 0, 0).is_unicast_site_local(), true);
    /// }
    /// ```
    pub fn is_unicast_site_local(&self) -> bool {
        (self.segments()[0] & 0xffc0) == 0xfec0
    }

    /// Returns true if this is an address reserved for documentation
    /// (2001:db8::/32).
    ///
    /// This property is defined in [RFC 3849].
    ///
    /// [RFC 3849]: https://tools.ietf.org/html/rfc3849
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// fn main() {
    ///     assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_documentation(),
    ///                false);
    ///     assert_eq!(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0).is_documentation(), true);
    /// }
    /// ```
    pub fn is_documentation(&self) -> bool {
        (self.segments()[0] == 0x2001) && (self.segments()[1] == 0xdb8)
    }

    /// Returns true if the address is a globally routable unicast address.
    ///
    /// The following return false:
    ///
    /// - the loopback address
    /// - the link-local addresses
    /// - the (deprecated) site-local addresses
    /// - unique local addresses
    /// - the unspecified address
    /// - the address range reserved for documentation
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// fn main() {
    ///     assert_eq!(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0).is_unicast_global(), false);
    ///     assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_unicast_global(),
    ///                true);
    /// }
    /// ```
    pub fn is_unicast_global(&self) -> bool {
        !self.is_multicast()
            && !self.is_loopback() && !self.is_unicast_link_local()
            && !self.is_unicast_site_local() && !self.is_unique_local()
            && !self.is_unspecified() && !self.is_documentation()
    }

    /// Returns the address's multicast scope if the address is multicast.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::{Ipv6Addr, Ipv6MulticastScope};
    ///
    /// fn main() {
    ///     assert_eq!(Ipv6Addr::new(0xff0e, 0, 0, 0, 0, 0, 0, 0).multicast_scope(),
    ///                              Some(Ipv6MulticastScope::Global));
    ///     assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).multicast_scope(), None);
    /// }
    /// ```
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

    /// Returns true if this is a multicast address (ff00::/8).
    ///
    /// This property is defined by [RFC 4291].
    ///
    /// [RFC 4291]: https://tools.ietf.org/html/rfc4291
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0).is_multicast(), true);
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_multicast(), false);
    /// ```
    #[stable(since = "1.7.0", feature = "ip_17")]
    pub fn is_multicast(&self) -> bool {
        (self.segments()[0] & 0xff00) == 0xff00
    }

    /// Converts this address to an IPv4 address. Returns None if this address is
    /// neither IPv4-compatible or IPv4-mapped.
    ///
    /// ::a.b.c.d and ::ffff:a.b.c.d become a.b.c.d
    ///
    /// ```
    /// use std::net::{Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0).to_ipv4(), None);
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).to_ipv4(),
    ///            Some(Ipv4Addr::new(192, 10, 2, 255)));
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1).to_ipv4(),
    ///            Some(Ipv4Addr::new(0, 0, 0, 1)));
    /// ```
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

    /// Returns the sixteen eight-bit integers the IPv6 address consists of.
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0).octets(),
    ///            [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    /// ```
    #[stable(feature = "ipv6_to_octets", since = "1.12.0")]
    pub fn octets(&self) -> [u8; 16] {
        self.inner.s6_addr
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
                    fn fmt_subslice(segments: &[u16], fmt: &mut fmt::Formatter) -> fmt::Result {
                        if !segments.is_empty() {
                            write!(fmt, "{:x}", segments[0])?;
                            for &seg in &segments[1..] {
                                write!(fmt, ":{:x}", seg)?;
                            }
                        }
                        Ok(())
                    }

                    fmt_subslice(&self.segments()[..zeros_at], fmt)?;
                    fmt.write_str("::")?;
                    fmt_subslice(&self.segments()[zeros_at + zeros_len..], fmt)
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

#[stable(feature = "ip_cmp", since = "1.15.0")]
impl PartialEq<IpAddr> for Ipv6Addr {
    fn eq(&self, other: &IpAddr) -> bool {
        match *other {
            IpAddr::V4(_) => false,
            IpAddr::V6(ref v6) => self == v6,
        }
    }
}

#[stable(feature = "ip_cmp", since = "1.15.0")]
impl PartialEq<Ipv6Addr> for IpAddr {
    fn eq(&self, other: &Ipv6Addr) -> bool {
        match *self {
            IpAddr::V4(_) => false,
            IpAddr::V6(ref v6) => v6 == other,
        }
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

#[stable(feature = "ip_cmp", since = "1.15.0")]
impl PartialOrd<Ipv6Addr> for IpAddr {
    fn partial_cmp(&self, other: &Ipv6Addr) -> Option<Ordering> {
        match *self {
            IpAddr::V4(_) => Some(Ordering::Less),
            IpAddr::V6(ref v6) => v6.partial_cmp(other),
        }
    }
}

#[stable(feature = "ip_cmp", since = "1.15.0")]
impl PartialOrd<IpAddr> for Ipv6Addr {
    fn partial_cmp(&self, other: &IpAddr) -> Option<Ordering> {
        match *other {
            IpAddr::V4(_) => Some(Ordering::Greater),
            IpAddr::V6(ref v6) => self.partial_cmp(v6),
        }
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

#[stable(feature = "ipv6_from_octets", since = "1.9.0")]
impl From<[u8; 16]> for Ipv6Addr {
    fn from(octets: [u8; 16]) -> Ipv6Addr {
        let mut inner: c::in6_addr = unsafe { mem::zeroed() };
        inner.s6_addr = octets;
        Ipv6Addr::from_inner(inner)
    }
}

#[stable(feature = "ipv6_from_segments", since = "1.15.0")]
impl From<[u16; 8]> for Ipv6Addr {
    fn from(segments: [u16; 8]) -> Ipv6Addr {
        let [a, b, c, d, e, f, g, h] = segments;
        Ipv6Addr::new(a, b, c, d, e, f, g, h)
    }
}

// Tests for this module
#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
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
    fn ip_properties() {
        fn check4(octets: &[u8; 4], unspec: bool, loopback: bool,
                  global: bool, multicast: bool, documentation: bool) {
            let ip = IpAddr::V4(Ipv4Addr::new(octets[0], octets[1], octets[2], octets[3]));
            assert_eq!(ip.is_unspecified(), unspec);
            assert_eq!(ip.is_loopback(), loopback);
            assert_eq!(ip.is_global(), global);
            assert_eq!(ip.is_multicast(), multicast);
            assert_eq!(ip.is_documentation(), documentation);
        }

        fn check6(str_addr: &str, unspec: bool, loopback: bool,
                  global: bool, u_doc: bool, mcast: bool) {
            let ip = IpAddr::V6(str_addr.parse().unwrap());
            assert_eq!(ip.is_unspecified(), unspec);
            assert_eq!(ip.is_loopback(), loopback);
            assert_eq!(ip.is_global(), global);
            assert_eq!(ip.is_documentation(), u_doc);
            assert_eq!(ip.is_multicast(), mcast);
        }

        //     address                unspec loopbk global multicast doc
        check4(&[0, 0, 0, 0],         true,  false, false,  false,   false);
        check4(&[0, 0, 0, 1],         false, false, true,   false,   false);
        check4(&[0, 1, 0, 0],         false, false, true,   false,   false);
        check4(&[10, 9, 8, 7],        false, false, false,  false,   false);
        check4(&[127, 1, 2, 3],       false, true,  false,  false,   false);
        check4(&[172, 31, 254, 253],  false, false, false,  false,   false);
        check4(&[169, 254, 253, 242], false, false, false,  false,   false);
        check4(&[192, 0, 2, 183],     false, false, false,  false,   true);
        check4(&[192, 1, 2, 183],     false, false, true,   false,   false);
        check4(&[192, 168, 254, 253], false, false, false,  false,   false);
        check4(&[198, 51, 100, 0],    false, false, false,  false,   true);
        check4(&[203, 0, 113, 0],     false, false, false,  false,   true);
        check4(&[203, 2, 113, 0],     false, false, true,   false,   false);
        check4(&[224, 0, 0, 0],       false, false, true,   true,    false);
        check4(&[239, 255, 255, 255], false, false, true,   true,    false);
        check4(&[255, 255, 255, 255], false, false, false,  false,   false);

        //     address                            unspec loopbk global doc    mcast
        check6("::",                              true,  false, false, false, false);
        check6("::1",                             false, true,  false, false, false);
        check6("::0.0.0.2",                       false, false, true,  false, false);
        check6("1::",                             false, false, true,  false, false);
        check6("fc00::",                          false, false, false, false, false);
        check6("fdff:ffff::",                     false, false, false, false, false);
        check6("fe80:ffff::",                     false, false, false, false, false);
        check6("febf:ffff::",                     false, false, false, false, false);
        check6("fec0::",                          false, false, false, false, false);
        check6("ff01::",                          false, false, false, false, true);
        check6("ff02::",                          false, false, false, false, true);
        check6("ff03::",                          false, false, false, false, true);
        check6("ff04::",                          false, false, false, false, true);
        check6("ff05::",                          false, false, false, false, true);
        check6("ff08::",                          false, false, false, false, true);
        check6("ff0e::",                          false, false, true,  false, true);
        check6("2001:db8:85a3::8a2e:370:7334",    false, false, false, true,  false);
        check6("102:304:506:708:90a:b0c:d0e:f10", false, false, true,  false, false);
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
        check(&[0, 0, 0, 0],         true,  false, false, false, false,  false,    false,  false);
        check(&[0, 0, 0, 1],         false, false, false, false, true,   false,    false,  false);
        check(&[0, 1, 0, 0],         false, false, false, false, true,   false,    false,  false);
        check(&[10, 9, 8, 7],        false, false, true,  false, false,  false,    false,  false);
        check(&[127, 1, 2, 3],       false, true,  false, false, false,  false,    false,  false);
        check(&[172, 31, 254, 253],  false, false, true,  false, false,  false,    false,  false);
        check(&[169, 254, 253, 242], false, false, false, true,  false,  false,    false,  false);
        check(&[192, 0, 2, 183],     false, false, false, false, false,  false,    false,  true);
        check(&[192, 1, 2, 183],     false, false, false, false, true,   false,    false,  false);
        check(&[192, 168, 254, 253], false, false, true,  false, false,  false,    false,  false);
        check(&[198, 51, 100, 0],    false, false, false, false, false,  false,    false,  true);
        check(&[203, 0, 113, 0],     false, false, false, false, false,  false,    false,  true);
        check(&[203, 2, 113, 0],     false, false, false, false, true,   false,    false,  false);
        check(&[224, 0, 0, 0],       false, false, false, false, true,   true,     false,  false);
        check(&[239, 255, 255, 255], false, false, false, false, true,   true,     false,  false);
        check(&[255, 255, 255, 255], false, false, false, false, false,  false,    true,   false);
    }

    #[test]
    fn ipv6_properties() {
        fn check(str_addr: &str, octets: &[u8; 16], unspec: bool, loopback: bool,
                 unique_local: bool, global: bool,
                 u_link_local: bool, u_site_local: bool, u_global: bool, u_doc: bool,
                 m_scope: Option<Ipv6MulticastScope>) {
            let ip: Ipv6Addr = str_addr.parse().unwrap();
            assert_eq!(str_addr, ip.to_string());
            assert_eq!(&ip.octets(), octets);
            assert_eq!(Ipv6Addr::from(*octets), ip);

            assert_eq!(ip.is_unspecified(), unspec);
            assert_eq!(ip.is_loopback(), loopback);
            assert_eq!(ip.is_unique_local(), unique_local);
            assert_eq!(ip.is_global(), global);
            assert_eq!(ip.is_unicast_link_local(), u_link_local);
            assert_eq!(ip.is_unicast_site_local(), u_site_local);
            assert_eq!(ip.is_unicast_global(), u_global);
            assert_eq!(ip.is_documentation(), u_doc);
            assert_eq!(ip.multicast_scope(), m_scope);
            assert_eq!(ip.is_multicast(), m_scope.is_some());
        }

        //    unspec loopbk uniqlo global unill  unisl  uniglo doc    mscope
        check("::", &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              true,  false, false, false, false, false, false, false, None);
        check("::1", &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              false, true,  false, false, false, false, false, false, None);
        check("::0.0.0.2", &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
              false, false, false, true,  false, false, true,  false, None);
        check("1::", &[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, false, true,  false, false, true,  false, None);
        check("fc00::", &[0xfc, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, true,  false, false, false, false, false, None);
        check("fdff:ffff::", &[0xfd, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, true,  false, false, false, false, false, None);
        check("fe80:ffff::", &[0xfe, 0x80, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, false, false, true,  false, false, false, None);
        check("febf:ffff::", &[0xfe, 0xbf, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, false, false, true,  false, false, false, None);
        check("fec0::", &[0xfe, 0xc0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, false, false, false, true,  false, false, None);
        check("ff01::", &[0xff, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, false, false, false, false, false, false, Some(InterfaceLocal));
        check("ff02::", &[0xff, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, false, false, false, false, false, false, Some(LinkLocal));
        check("ff03::", &[0xff, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, false, false, false, false, false, false, Some(RealmLocal));
        check("ff04::", &[0xff, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, false, false, false, false, false, false, Some(AdminLocal));
        check("ff05::", &[0xff, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, false, false, false, false, false, false, Some(SiteLocal));
        check("ff08::", &[0xff, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, false, false, false, false, false, false, Some(OrganizationLocal));
        check("ff0e::", &[0xff, 0xe, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              false, false, false, true,  false, false, false, false, Some(Global));
        check("2001:db8:85a3::8a2e:370:7334",
              &[0x20, 1, 0xd, 0xb8, 0x85, 0xa3, 0, 0, 0, 0, 0x8a, 0x2e, 3, 0x70, 0x73, 0x34],
              false, false, false, false, false, false, false, true, None);
        check("102:304:506:708:90a:b0c:d0e:f10",
              &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
              false, false, false, true,  false, false, true,  false, None);
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
    fn ipv4_from_octets() {
        assert_eq!(Ipv4Addr::from([127, 0, 0, 1]), Ipv4Addr::new(127, 0, 0, 1))
    }

    #[test]
    fn ipv6_from_segments() {
        let from_u16s = Ipv6Addr::from([0x0011, 0x2233, 0x4455, 0x6677,
                                        0x8899, 0xaabb, 0xccdd, 0xeeff]);
        let new = Ipv6Addr::new(0x0011, 0x2233, 0x4455, 0x6677,
                                0x8899, 0xaabb, 0xccdd, 0xeeff);
        assert_eq!(new, from_u16s);
    }

    #[test]
    fn ipv6_from_octets() {
        let from_u16s = Ipv6Addr::from([0x0011, 0x2233, 0x4455, 0x6677,
                                        0x8899, 0xaabb, 0xccdd, 0xeeff]);
        let from_u8s = Ipv6Addr::from([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                                       0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]);
        assert_eq!(from_u16s, from_u8s);
    }

    #[test]
    fn cmp() {
        let v41 = Ipv4Addr::new(100, 64, 3, 3);
        let v42 = Ipv4Addr::new(192, 0, 2, 2);
        let v61 = "2001:db8:f00::1002".parse::<Ipv6Addr>().unwrap();
        let v62 = "2001:db8:f00::2001".parse::<Ipv6Addr>().unwrap();
        assert!(v41 < v42);
        assert!(v61 < v62);

        assert_eq!(v41, IpAddr::V4(v41));
        assert_eq!(v61, IpAddr::V6(v61));
        assert!(v41 != IpAddr::V4(v42));
        assert!(v61 != IpAddr::V6(v62));

        assert!(v41 < IpAddr::V4(v42));
        assert!(v61 < IpAddr::V6(v62));
        assert!(IpAddr::V4(v41) < v42);
        assert!(IpAddr::V6(v61) < v62);

        assert!(v41 < IpAddr::V6(v61));
        assert!(IpAddr::V4(v41) < v61);
    }

    #[test]
    fn is_v4() {
        let ip = IpAddr::V4(Ipv4Addr::new(100, 64, 3, 3));
        assert!(ip.is_ipv4());
        assert!(!ip.is_ipv6());
    }

    #[test]
    fn is_v6() {
        let ip = IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x1234, 0x5678));
        assert!(!ip.is_ipv4());
        assert!(ip.is_ipv6());
    }
}
