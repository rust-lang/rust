use super::display_buffer::DisplayBuffer;
use crate::cmp::Ordering;
use crate::fmt::{self, Write};
use crate::hash::{Hash, Hasher};
use crate::mem::transmute;
use crate::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

/// An IP address, either IPv4 or IPv6.
///
/// This enum can contain either an [`Ipv4Addr`] or an [`Ipv6Addr`], see their
/// respective documentation for more details.
///
/// # Examples
///
/// ```
/// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
///
/// let localhost_v4 = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
/// let localhost_v6 = IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1));
///
/// assert_eq!("127.0.0.1".parse(), Ok(localhost_v4));
/// assert_eq!("::1".parse(), Ok(localhost_v6));
///
/// assert_eq!(localhost_v4.is_ipv6(), false);
/// assert_eq!(localhost_v4.is_ipv4(), true);
/// ```
#[rustc_diagnostic_item = "IpAddr"]
#[stable(feature = "ip_addr", since = "1.7.0")]
#[derive(Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum IpAddr {
    /// An IPv4 address.
    #[stable(feature = "ip_addr", since = "1.7.0")]
    V4(#[stable(feature = "ip_addr", since = "1.7.0")] Ipv4Addr),
    /// An IPv6 address.
    #[stable(feature = "ip_addr", since = "1.7.0")]
    V6(#[stable(feature = "ip_addr", since = "1.7.0")] Ipv6Addr),
}

/// An IPv4 address.
///
/// IPv4 addresses are defined as 32-bit integers in [IETF RFC 791].
/// They are usually represented as four octets.
///
/// See [`IpAddr`] for a type encompassing both IPv4 and IPv6 addresses.
///
/// [IETF RFC 791]: https://tools.ietf.org/html/rfc791
///
/// # Textual representation
///
/// `Ipv4Addr` provides a [`FromStr`] implementation. The four octets are in decimal
/// notation, divided by `.` (this is called "dot-decimal notation").
/// Notably, octal numbers (which are indicated with a leading `0`) and hexadecimal numbers (which
/// are indicated with a leading `0x`) are not allowed per [IETF RFC 6943].
///
/// [IETF RFC 6943]: https://tools.ietf.org/html/rfc6943#section-3.1.1
/// [`FromStr`]: crate::str::FromStr
///
/// # Examples
///
/// ```
/// use std::net::Ipv4Addr;
///
/// let localhost = Ipv4Addr::new(127, 0, 0, 1);
/// assert_eq!("127.0.0.1".parse(), Ok(localhost));
/// assert_eq!(localhost.is_loopback(), true);
/// assert!("012.004.002.000".parse::<Ipv4Addr>().is_err()); // all octets are in octal
/// assert!("0000000.0.0.0".parse::<Ipv4Addr>().is_err()); // first octet is a zero in octal
/// assert!("0xcb.0x0.0x71.0x00".parse::<Ipv4Addr>().is_err()); // all octets are in hex
/// ```
#[rustc_diagnostic_item = "Ipv4Addr"]
#[derive(Copy, Clone, PartialEq, Eq)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Ipv4Addr {
    octets: [u8; 4],
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for Ipv4Addr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hashers are often more efficient at hashing a fixed-width integer
        // than a bytestring, so convert before hashing. We don't use to_bits()
        // here as that may involve a byteswap which is unnecessary.
        u32::from_ne_bytes(self.octets).hash(state);
    }
}

/// An IPv6 address.
///
/// IPv6 addresses are defined as 128-bit integers in [IETF RFC 4291].
/// They are usually represented as eight 16-bit segments.
///
/// [IETF RFC 4291]: https://tools.ietf.org/html/rfc4291
///
/// # Embedding IPv4 Addresses
///
/// See [`IpAddr`] for a type encompassing both IPv4 and IPv6 addresses.
///
/// To assist in the transition from IPv4 to IPv6 two types of IPv6 addresses that embed an IPv4 address were defined:
/// IPv4-compatible and IPv4-mapped addresses. Of these IPv4-compatible addresses have been officially deprecated.
///
/// Both types of addresses are not assigned any special meaning by this implementation,
/// other than what the relevant standards prescribe. This means that an address like `::ffff:127.0.0.1`,
/// while representing an IPv4 loopback address, is not itself an IPv6 loopback address; only `::1` is.
/// To handle these so called "IPv4-in-IPv6" addresses, they have to first be converted to their canonical IPv4 address.
///
/// ### IPv4-Compatible IPv6 Addresses
///
/// IPv4-compatible IPv6 addresses are defined in [IETF RFC 4291 Section 2.5.5.1], and have been officially deprecated.
/// The RFC describes the format of an "IPv4-Compatible IPv6 address" as follows:
///
/// ```text
/// |                80 bits               | 16 |      32 bits        |
/// +--------------------------------------+--------------------------+
/// |0000..............................0000|0000|    IPv4 address     |
/// +--------------------------------------+----+---------------------+
/// ```
/// So `::a.b.c.d` would be an IPv4-compatible IPv6 address representing the IPv4 address `a.b.c.d`.
///
/// To convert from an IPv4 address to an IPv4-compatible IPv6 address, use [`Ipv4Addr::to_ipv6_compatible`].
/// Use [`Ipv6Addr::to_ipv4`] to convert an IPv4-compatible IPv6 address to the canonical IPv4 address.
///
/// [IETF RFC 4291 Section 2.5.5.1]: https://datatracker.ietf.org/doc/html/rfc4291#section-2.5.5.1
///
/// ### IPv4-Mapped IPv6 Addresses
///
/// IPv4-mapped IPv6 addresses are defined in [IETF RFC 4291 Section 2.5.5.2].
/// The RFC describes the format of an "IPv4-Mapped IPv6 address" as follows:
///
/// ```text
/// |                80 bits               | 16 |      32 bits        |
/// +--------------------------------------+--------------------------+
/// |0000..............................0000|FFFF|    IPv4 address     |
/// +--------------------------------------+----+---------------------+
/// ```
/// So `::ffff:a.b.c.d` would be an IPv4-mapped IPv6 address representing the IPv4 address `a.b.c.d`.
///
/// To convert from an IPv4 address to an IPv4-mapped IPv6 address, use [`Ipv4Addr::to_ipv6_mapped`].
/// Use [`Ipv6Addr::to_ipv4`] to convert an IPv4-mapped IPv6 address to the canonical IPv4 address.
/// Note that this will also convert the IPv6 loopback address `::1` to `0.0.0.1`. Use
/// [`Ipv6Addr::to_ipv4_mapped`] to avoid this.
///
/// [IETF RFC 4291 Section 2.5.5.2]: https://datatracker.ietf.org/doc/html/rfc4291#section-2.5.5.2
///
/// # Textual representation
///
/// `Ipv6Addr` provides a [`FromStr`] implementation. There are many ways to represent
/// an IPv6 address in text, but in general, each segments is written in hexadecimal
/// notation, and segments are separated by `:`. For more information, see
/// [IETF RFC 5952].
///
/// [`FromStr`]: crate::str::FromStr
/// [IETF RFC 5952]: https://tools.ietf.org/html/rfc5952
///
/// # Examples
///
/// ```
/// use std::net::Ipv6Addr;
///
/// let localhost = Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1);
/// assert_eq!("::1".parse(), Ok(localhost));
/// assert_eq!(localhost.is_loopback(), true);
/// ```
#[rustc_diagnostic_item = "Ipv6Addr"]
#[derive(Copy, Clone, PartialEq, Eq)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Ipv6Addr {
    octets: [u8; 16],
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for Ipv6Addr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hashers are often more efficient at hashing a fixed-width integer
        // than a bytestring, so convert before hashing. We don't use to_bits()
        // here as that may involve unnecessary byteswaps.
        u128::from_ne_bytes(self.octets).hash(state);
    }
}

/// Scope of an [IPv6 multicast address] as defined in [IETF RFC 7346 section 2].
///
/// # Stability Guarantees
///
/// Not all possible values for a multicast scope have been assigned.
/// Future RFCs may introduce new scopes, which will be added as variants to this enum;
/// because of this the enum is marked as `#[non_exhaustive]`.
///
/// # Examples
/// ```
/// #![feature(ip)]
///
/// use std::net::Ipv6Addr;
/// use std::net::Ipv6MulticastScope::*;
///
/// // An IPv6 multicast address with global scope (`ff0e::`).
/// let address = Ipv6Addr::new(0xff0e, 0, 0, 0, 0, 0, 0, 0);
///
/// // Will print "Global scope".
/// match address.multicast_scope() {
///     Some(InterfaceLocal) => println!("Interface-Local scope"),
///     Some(LinkLocal) => println!("Link-Local scope"),
///     Some(RealmLocal) => println!("Realm-Local scope"),
///     Some(AdminLocal) => println!("Admin-Local scope"),
///     Some(SiteLocal) => println!("Site-Local scope"),
///     Some(OrganizationLocal) => println!("Organization-Local scope"),
///     Some(Global) => println!("Global scope"),
///     Some(_) => println!("Unknown scope"),
///     None => println!("Not a multicast address!")
/// }
///
/// ```
///
/// [IPv6 multicast address]: Ipv6Addr
/// [IETF RFC 7346 section 2]: https://tools.ietf.org/html/rfc7346#section-2
#[derive(Copy, PartialEq, Eq, Clone, Hash, Debug)]
#[unstable(feature = "ip", issue = "27709")]
#[non_exhaustive]
pub enum Ipv6MulticastScope {
    /// Interface-Local scope.
    InterfaceLocal,
    /// Link-Local scope.
    LinkLocal,
    /// Realm-Local scope.
    RealmLocal,
    /// Admin-Local scope.
    AdminLocal,
    /// Site-Local scope.
    SiteLocal,
    /// Organization-Local scope.
    OrganizationLocal,
    /// Global scope.
    Global,
}

impl IpAddr {
    /// Returns [`true`] for the special 'unspecified' address.
    ///
    /// See the documentation for [`Ipv4Addr::is_unspecified()`] and
    /// [`Ipv6Addr::is_unspecified()`] for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)).is_unspecified(), true);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0)).is_unspecified(), true);
    /// ```
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(feature = "ip_shared", since = "1.12.0")]
    #[must_use]
    #[inline]
    pub const fn is_unspecified(&self) -> bool {
        match self {
            IpAddr::V4(ip) => ip.is_unspecified(),
            IpAddr::V6(ip) => ip.is_unspecified(),
        }
    }

    /// Returns [`true`] if this is a loopback address.
    ///
    /// See the documentation for [`Ipv4Addr::is_loopback()`] and
    /// [`Ipv6Addr::is_loopback()`] for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)).is_loopback(), true);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0x1)).is_loopback(), true);
    /// ```
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(feature = "ip_shared", since = "1.12.0")]
    #[must_use]
    #[inline]
    pub const fn is_loopback(&self) -> bool {
        match self {
            IpAddr::V4(ip) => ip.is_loopback(),
            IpAddr::V6(ip) => ip.is_loopback(),
        }
    }

    /// Returns [`true`] if the address appears to be globally routable.
    ///
    /// See the documentation for [`Ipv4Addr::is_global()`] and
    /// [`Ipv6Addr::is_global()`] for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(80, 9, 12, 3)).is_global(), true);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::new(0, 0, 0x1c9, 0, 0, 0xafc8, 0, 0x1)).is_global(), true);
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_global(&self) -> bool {
        match self {
            IpAddr::V4(ip) => ip.is_global(),
            IpAddr::V6(ip) => ip.is_global(),
        }
    }

    /// Returns [`true`] if this is a multicast address.
    ///
    /// See the documentation for [`Ipv4Addr::is_multicast()`] and
    /// [`Ipv6Addr::is_multicast()`] for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(224, 254, 0, 0)).is_multicast(), true);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0)).is_multicast(), true);
    /// ```
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(feature = "ip_shared", since = "1.12.0")]
    #[must_use]
    #[inline]
    pub const fn is_multicast(&self) -> bool {
        match self {
            IpAddr::V4(ip) => ip.is_multicast(),
            IpAddr::V6(ip) => ip.is_multicast(),
        }
    }

    /// Returns [`true`] if this address is in a range designated for documentation.
    ///
    /// See the documentation for [`Ipv4Addr::is_documentation()`] and
    /// [`Ipv6Addr::is_documentation()`] for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 6)).is_documentation(), true);
    /// assert_eq!(
    ///     IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0)).is_documentation(),
    ///     true
    /// );
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_documentation(&self) -> bool {
        match self {
            IpAddr::V4(ip) => ip.is_documentation(),
            IpAddr::V6(ip) => ip.is_documentation(),
        }
    }

    /// Returns [`true`] if this address is in a range designated for benchmarking.
    ///
    /// See the documentation for [`Ipv4Addr::is_benchmarking()`] and
    /// [`Ipv6Addr::is_benchmarking()`] for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(198, 19, 255, 255)).is_benchmarking(), true);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::new(0x2001, 0x2, 0, 0, 0, 0, 0, 0)).is_benchmarking(), true);
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_benchmarking(&self) -> bool {
        match self {
            IpAddr::V4(ip) => ip.is_benchmarking(),
            IpAddr::V6(ip) => ip.is_benchmarking(),
        }
    }

    /// Returns [`true`] if this address is an [`IPv4` address], and [`false`]
    /// otherwise.
    ///
    /// [`IPv4` address]: IpAddr::V4
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 6)).is_ipv4(), true);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0)).is_ipv4(), false);
    /// ```
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(feature = "ipaddr_checker", since = "1.16.0")]
    #[must_use]
    #[inline]
    pub const fn is_ipv4(&self) -> bool {
        matches!(self, IpAddr::V4(_))
    }

    /// Returns [`true`] if this address is an [`IPv6` address], and [`false`]
    /// otherwise.
    ///
    /// [`IPv6` address]: IpAddr::V6
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 6)).is_ipv6(), false);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0)).is_ipv6(), true);
    /// ```
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(feature = "ipaddr_checker", since = "1.16.0")]
    #[must_use]
    #[inline]
    pub const fn is_ipv6(&self) -> bool {
        matches!(self, IpAddr::V6(_))
    }

    /// Converts this address to an `IpAddr::V4` if it is an IPv4-mapped IPv6
    /// address, otherwise returns `self` as-is.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// let localhost_v4 = Ipv4Addr::new(127, 0, 0, 1);
    ///
    /// assert_eq!(IpAddr::V4(localhost_v4).to_canonical(), localhost_v4);
    /// assert_eq!(IpAddr::V6(localhost_v4.to_ipv6_mapped()).to_canonical(), localhost_v4);
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)).to_canonical().is_loopback(), true);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x7f00, 0x1)).is_loopback(), false);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x7f00, 0x1)).to_canonical().is_loopback(), true);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "ip_to_canonical", since = "1.75.0")]
    #[rustc_const_stable(feature = "ip_to_canonical", since = "1.75.0")]
    pub const fn to_canonical(&self) -> IpAddr {
        match self {
            IpAddr::V4(_) => *self,
            IpAddr::V6(v6) => v6.to_canonical(),
        }
    }

    /// Returns the eight-bit integers this address consists of as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip_as_octets)]
    ///
    /// use std::net::{Ipv4Addr, Ipv6Addr, IpAddr};
    ///
    /// assert_eq!(IpAddr::V4(Ipv4Addr::LOCALHOST).as_octets(), &[127, 0, 0, 1]);
    /// assert_eq!(IpAddr::V6(Ipv6Addr::LOCALHOST).as_octets(),
    ///            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    /// ```
    #[unstable(feature = "ip_as_octets", issue = "137259")]
    #[inline]
    pub const fn as_octets(&self) -> &[u8] {
        match self {
            IpAddr::V4(ip) => ip.as_octets().as_slice(),
            IpAddr::V6(ip) => ip.as_octets().as_slice(),
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
    #[rustc_const_stable(feature = "const_ip_32", since = "1.32.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    pub const fn new(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr {
        Ipv4Addr { octets: [a, b, c, d] }
    }

    /// The size of an IPv4 address in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::BITS, 32);
    /// ```
    #[stable(feature = "ip_bits", since = "1.80.0")]
    pub const BITS: u32 = 32;

    /// Converts an IPv4 address into a `u32` representation using native byte order.
    ///
    /// Although IPv4 addresses are big-endian, the `u32` value will use the target platform's
    /// native byte order. That is, the `u32` value is an integer representation of the IPv4
    /// address and not an integer interpretation of the IPv4 address's big-endian bitstring. This
    /// means that the `u32` value masked with `0xffffff00` will set the last octet in the address
    /// to 0, regardless of the target platform's endianness.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// let addr = Ipv4Addr::new(0x12, 0x34, 0x56, 0x78);
    /// assert_eq!(0x12345678, addr.to_bits());
    /// ```
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// let addr = Ipv4Addr::new(0x12, 0x34, 0x56, 0x78);
    /// let addr_bits = addr.to_bits() & 0xffffff00;
    /// assert_eq!(Ipv4Addr::new(0x12, 0x34, 0x56, 0x00), Ipv4Addr::from_bits(addr_bits));
    ///
    /// ```
    #[rustc_const_stable(feature = "ip_bits", since = "1.80.0")]
    #[stable(feature = "ip_bits", since = "1.80.0")]
    #[must_use]
    #[inline]
    pub const fn to_bits(self) -> u32 {
        u32::from_be_bytes(self.octets)
    }

    /// Converts a native byte order `u32` into an IPv4 address.
    ///
    /// See [`Ipv4Addr::to_bits`] for an explanation on endianness.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// let addr = Ipv4Addr::from_bits(0x12345678);
    /// assert_eq!(Ipv4Addr::new(0x12, 0x34, 0x56, 0x78), addr);
    /// ```
    #[rustc_const_stable(feature = "ip_bits", since = "1.80.0")]
    #[stable(feature = "ip_bits", since = "1.80.0")]
    #[must_use]
    #[inline]
    pub const fn from_bits(bits: u32) -> Ipv4Addr {
        Ipv4Addr { octets: bits.to_be_bytes() }
    }

    /// An IPv4 address with the address pointing to localhost: `127.0.0.1`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// let addr = Ipv4Addr::LOCALHOST;
    /// assert_eq!(addr, Ipv4Addr::new(127, 0, 0, 1));
    /// ```
    #[stable(feature = "ip_constructors", since = "1.30.0")]
    pub const LOCALHOST: Self = Ipv4Addr::new(127, 0, 0, 1);

    /// An IPv4 address representing an unspecified address: `0.0.0.0`
    ///
    /// This corresponds to the constant `INADDR_ANY` in other languages.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// let addr = Ipv4Addr::UNSPECIFIED;
    /// assert_eq!(addr, Ipv4Addr::new(0, 0, 0, 0));
    /// ```
    #[doc(alias = "INADDR_ANY")]
    #[stable(feature = "ip_constructors", since = "1.30.0")]
    pub const UNSPECIFIED: Self = Ipv4Addr::new(0, 0, 0, 0);

    /// An IPv4 address representing the broadcast address: `255.255.255.255`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// let addr = Ipv4Addr::BROADCAST;
    /// assert_eq!(addr, Ipv4Addr::new(255, 255, 255, 255));
    /// ```
    #[stable(feature = "ip_constructors", since = "1.30.0")]
    pub const BROADCAST: Self = Ipv4Addr::new(255, 255, 255, 255);

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
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    pub const fn octets(&self) -> [u8; 4] {
        self.octets
    }

    /// Creates an `Ipv4Addr` from a four element byte array.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// let addr = Ipv4Addr::from_octets([13u8, 12u8, 11u8, 10u8]);
    /// assert_eq!(Ipv4Addr::new(13, 12, 11, 10), addr);
    /// ```
    #[stable(feature = "ip_from", since = "1.91.0")]
    #[rustc_const_stable(feature = "ip_from", since = "1.91.0")]
    #[must_use]
    #[inline]
    pub const fn from_octets(octets: [u8; 4]) -> Ipv4Addr {
        Ipv4Addr { octets }
    }

    /// Returns the four eight-bit integers that make up this address
    /// as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip_as_octets)]
    ///
    /// use std::net::Ipv4Addr;
    ///
    /// let addr = Ipv4Addr::new(127, 0, 0, 1);
    /// assert_eq!(addr.as_octets(), &[127, 0, 0, 1]);
    /// ```
    #[unstable(feature = "ip_as_octets", issue = "137259")]
    #[inline]
    pub const fn as_octets(&self) -> &[u8; 4] {
        &self.octets
    }

    /// Returns [`true`] for the special 'unspecified' address (`0.0.0.0`).
    ///
    /// This property is defined in _UNIX Network Programming, Second Edition_,
    /// W. Richard Stevens, p. 891; see also [ip7].
    ///
    /// [ip7]: https://man7.org/linux/man-pages/man7/ip.7.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(0, 0, 0, 0).is_unspecified(), true);
    /// assert_eq!(Ipv4Addr::new(45, 22, 13, 197).is_unspecified(), false);
    /// ```
    #[rustc_const_stable(feature = "const_ip_32", since = "1.32.0")]
    #[stable(feature = "ip_shared", since = "1.12.0")]
    #[must_use]
    #[inline]
    pub const fn is_unspecified(&self) -> bool {
        u32::from_be_bytes(self.octets) == 0
    }

    /// Returns [`true`] if this is a loopback address (`127.0.0.0/8`).
    ///
    /// This property is defined by [IETF RFC 1122].
    ///
    /// [IETF RFC 1122]: https://tools.ietf.org/html/rfc1122
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(127, 0, 0, 1).is_loopback(), true);
    /// assert_eq!(Ipv4Addr::new(45, 22, 13, 197).is_loopback(), false);
    /// ```
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(since = "1.7.0", feature = "ip_17")]
    #[must_use]
    #[inline]
    pub const fn is_loopback(&self) -> bool {
        self.octets()[0] == 127
    }

    /// Returns [`true`] if this is a private address.
    ///
    /// The private address ranges are defined in [IETF RFC 1918] and include:
    ///
    ///  - `10.0.0.0/8`
    ///  - `172.16.0.0/12`
    ///  - `192.168.0.0/16`
    ///
    /// [IETF RFC 1918]: https://tools.ietf.org/html/rfc1918
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
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(since = "1.7.0", feature = "ip_17")]
    #[must_use]
    #[inline]
    pub const fn is_private(&self) -> bool {
        match self.octets() {
            [10, ..] => true,
            [172, b, ..] if b >= 16 && b <= 31 => true,
            [192, 168, ..] => true,
            _ => false,
        }
    }

    /// Returns [`true`] if the address is link-local (`169.254.0.0/16`).
    ///
    /// This property is defined by [IETF RFC 3927].
    ///
    /// [IETF RFC 3927]: https://tools.ietf.org/html/rfc3927
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
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(since = "1.7.0", feature = "ip_17")]
    #[must_use]
    #[inline]
    pub const fn is_link_local(&self) -> bool {
        matches!(self.octets(), [169, 254, ..])
    }

    /// Returns [`true`] if the address appears to be globally reachable
    /// as specified by the [IANA IPv4 Special-Purpose Address Registry].
    ///
    /// Whether or not an address is practically reachable will depend on your
    /// network configuration. Most IPv4 addresses are globally reachable, unless
    /// they are specifically defined as *not* globally reachable.
    ///
    /// Non-exhaustive list of notable addresses that are not globally reachable:
    ///
    /// - The [unspecified address] ([`is_unspecified`](Ipv4Addr::is_unspecified))
    /// - Addresses reserved for private use ([`is_private`](Ipv4Addr::is_private))
    /// - Addresses in the shared address space ([`is_shared`](Ipv4Addr::is_shared))
    /// - Loopback addresses ([`is_loopback`](Ipv4Addr::is_loopback))
    /// - Link-local addresses ([`is_link_local`](Ipv4Addr::is_link_local))
    /// - Addresses reserved for documentation ([`is_documentation`](Ipv4Addr::is_documentation))
    /// - Addresses reserved for benchmarking ([`is_benchmarking`](Ipv4Addr::is_benchmarking))
    /// - Reserved addresses ([`is_reserved`](Ipv4Addr::is_reserved))
    /// - The [broadcast address] ([`is_broadcast`](Ipv4Addr::is_broadcast))
    ///
    /// For the complete overview of which addresses are globally reachable, see the table at the [IANA IPv4 Special-Purpose Address Registry].
    ///
    /// [IANA IPv4 Special-Purpose Address Registry]: https://www.iana.org/assignments/iana-ipv4-special-registry/iana-ipv4-special-registry.xhtml
    /// [unspecified address]: Ipv4Addr::UNSPECIFIED
    /// [broadcast address]: Ipv4Addr::BROADCAST
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv4Addr;
    ///
    /// // Most IPv4 addresses are globally reachable:
    /// assert_eq!(Ipv4Addr::new(80, 9, 12, 3).is_global(), true);
    ///
    /// // However some addresses have been assigned a special meaning
    /// // that makes them not globally reachable. Some examples are:
    ///
    /// // The unspecified address (`0.0.0.0`)
    /// assert_eq!(Ipv4Addr::UNSPECIFIED.is_global(), false);
    ///
    /// // Addresses reserved for private use (`10.0.0.0/8`, `172.16.0.0/12`, 192.168.0.0/16)
    /// assert_eq!(Ipv4Addr::new(10, 254, 0, 0).is_global(), false);
    /// assert_eq!(Ipv4Addr::new(192, 168, 10, 65).is_global(), false);
    /// assert_eq!(Ipv4Addr::new(172, 16, 10, 65).is_global(), false);
    ///
    /// // Addresses in the shared address space (`100.64.0.0/10`)
    /// assert_eq!(Ipv4Addr::new(100, 100, 0, 0).is_global(), false);
    ///
    /// // The loopback addresses (`127.0.0.0/8`)
    /// assert_eq!(Ipv4Addr::LOCALHOST.is_global(), false);
    ///
    /// // Link-local addresses (`169.254.0.0/16`)
    /// assert_eq!(Ipv4Addr::new(169, 254, 45, 1).is_global(), false);
    ///
    /// // Addresses reserved for documentation (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`)
    /// assert_eq!(Ipv4Addr::new(192, 0, 2, 255).is_global(), false);
    /// assert_eq!(Ipv4Addr::new(198, 51, 100, 65).is_global(), false);
    /// assert_eq!(Ipv4Addr::new(203, 0, 113, 6).is_global(), false);
    ///
    /// // Addresses reserved for benchmarking (`198.18.0.0/15`)
    /// assert_eq!(Ipv4Addr::new(198, 18, 0, 0).is_global(), false);
    ///
    /// // Reserved addresses (`240.0.0.0/4`)
    /// assert_eq!(Ipv4Addr::new(250, 10, 20, 30).is_global(), false);
    ///
    /// // The broadcast address (`255.255.255.255`)
    /// assert_eq!(Ipv4Addr::BROADCAST.is_global(), false);
    ///
    /// // For a complete overview see the IANA IPv4 Special-Purpose Address Registry.
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_global(&self) -> bool {
        !(self.octets()[0] == 0 // "This network"
            || self.is_private()
            || self.is_shared()
            || self.is_loopback()
            || self.is_link_local()
            // addresses reserved for future protocols (`192.0.0.0/24`)
            // .9 and .10 are documented as globally reachable so they're excluded
            || (
                self.octets()[0] == 192 && self.octets()[1] == 0 && self.octets()[2] == 0
                && self.octets()[3] != 9 && self.octets()[3] != 10
            )
            || self.is_documentation()
            || self.is_benchmarking()
            || self.is_reserved()
            || self.is_broadcast())
    }

    /// Returns [`true`] if this address is part of the Shared Address Space defined in
    /// [IETF RFC 6598] (`100.64.0.0/10`).
    ///
    /// [IETF RFC 6598]: https://tools.ietf.org/html/rfc6598
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(100, 64, 0, 0).is_shared(), true);
    /// assert_eq!(Ipv4Addr::new(100, 127, 255, 255).is_shared(), true);
    /// assert_eq!(Ipv4Addr::new(100, 128, 0, 0).is_shared(), false);
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_shared(&self) -> bool {
        self.octets()[0] == 100 && (self.octets()[1] & 0b1100_0000 == 0b0100_0000)
    }

    /// Returns [`true`] if this address part of the `198.18.0.0/15` range, which is reserved for
    /// network devices benchmarking.
    ///
    /// This range is defined in [IETF RFC 2544] as `192.18.0.0` through
    /// `198.19.255.255` but [errata 423] corrects it to `198.18.0.0/15`.
    ///
    /// [IETF RFC 2544]: https://tools.ietf.org/html/rfc2544
    /// [errata 423]: https://www.rfc-editor.org/errata/eid423
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(198, 17, 255, 255).is_benchmarking(), false);
    /// assert_eq!(Ipv4Addr::new(198, 18, 0, 0).is_benchmarking(), true);
    /// assert_eq!(Ipv4Addr::new(198, 19, 255, 255).is_benchmarking(), true);
    /// assert_eq!(Ipv4Addr::new(198, 20, 0, 0).is_benchmarking(), false);
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_benchmarking(&self) -> bool {
        self.octets()[0] == 198 && (self.octets()[1] & 0xfe) == 18
    }

    /// Returns [`true`] if this address is reserved by IANA for future use.
    ///
    /// [IETF RFC 1112] defines the block of reserved addresses as `240.0.0.0/4`.
    /// This range normally includes the broadcast address `255.255.255.255`, but
    /// this implementation explicitly excludes it, since it is obviously not
    /// reserved for future use.
    ///
    /// [IETF RFC 1112]: https://tools.ietf.org/html/rfc1112
    ///
    /// # Warning
    ///
    /// As IANA assigns new addresses, this method will be
    /// updated. This may result in non-reserved addresses being
    /// treated as reserved in code that relies on an outdated version
    /// of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(240, 0, 0, 0).is_reserved(), true);
    /// assert_eq!(Ipv4Addr::new(255, 255, 255, 254).is_reserved(), true);
    ///
    /// assert_eq!(Ipv4Addr::new(239, 255, 255, 255).is_reserved(), false);
    /// // The broadcast address is not considered as reserved for future use by this implementation
    /// assert_eq!(Ipv4Addr::new(255, 255, 255, 255).is_reserved(), false);
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_reserved(&self) -> bool {
        self.octets()[0] & 240 == 240 && !self.is_broadcast()
    }

    /// Returns [`true`] if this is a multicast address (`224.0.0.0/4`).
    ///
    /// Multicast addresses have a most significant octet between `224` and `239`,
    /// and is defined by [IETF RFC 5771].
    ///
    /// [IETF RFC 5771]: https://tools.ietf.org/html/rfc5771
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
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(since = "1.7.0", feature = "ip_17")]
    #[must_use]
    #[inline]
    pub const fn is_multicast(&self) -> bool {
        self.octets()[0] >= 224 && self.octets()[0] <= 239
    }

    /// Returns [`true`] if this is a broadcast address (`255.255.255.255`).
    ///
    /// A broadcast address has all octets set to `255` as defined in [IETF RFC 919].
    ///
    /// [IETF RFC 919]: https://tools.ietf.org/html/rfc919
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// assert_eq!(Ipv4Addr::new(255, 255, 255, 255).is_broadcast(), true);
    /// assert_eq!(Ipv4Addr::new(236, 168, 10, 65).is_broadcast(), false);
    /// ```
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(since = "1.7.0", feature = "ip_17")]
    #[must_use]
    #[inline]
    pub const fn is_broadcast(&self) -> bool {
        u32::from_be_bytes(self.octets()) == u32::from_be_bytes(Self::BROADCAST.octets())
    }

    /// Returns [`true`] if this address is in a range designated for documentation.
    ///
    /// This is defined in [IETF RFC 5737]:
    ///
    /// - `192.0.2.0/24` (TEST-NET-1)
    /// - `198.51.100.0/24` (TEST-NET-2)
    /// - `203.0.113.0/24` (TEST-NET-3)
    ///
    /// [IETF RFC 5737]: https://tools.ietf.org/html/rfc5737
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
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(since = "1.7.0", feature = "ip_17")]
    #[must_use]
    #[inline]
    pub const fn is_documentation(&self) -> bool {
        matches!(self.octets(), [192, 0, 2, _] | [198, 51, 100, _] | [203, 0, 113, _])
    }

    /// Converts this address to an [IPv4-compatible] [`IPv6` address].
    ///
    /// `a.b.c.d` becomes `::a.b.c.d`
    ///
    /// Note that IPv4-compatible addresses have been officially deprecated.
    /// If you don't explicitly need an IPv4-compatible address for legacy reasons, consider using `to_ipv6_mapped` instead.
    ///
    /// [IPv4-compatible]: Ipv6Addr#ipv4-compatible-ipv6-addresses
    /// [`IPv6` address]: Ipv6Addr
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(
    ///     Ipv4Addr::new(192, 0, 2, 255).to_ipv6_compatible(),
    ///     Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0xc000, 0x2ff)
    /// );
    /// ```
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub const fn to_ipv6_compatible(&self) -> Ipv6Addr {
        let [a, b, c, d] = self.octets();
        Ipv6Addr { octets: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, c, d] }
    }

    /// Converts this address to an [IPv4-mapped] [`IPv6` address].
    ///
    /// `a.b.c.d` becomes `::ffff:a.b.c.d`
    ///
    /// [IPv4-mapped]: Ipv6Addr#ipv4-mapped-ipv6-addresses
    /// [`IPv6` address]: Ipv6Addr
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(Ipv4Addr::new(192, 0, 2, 255).to_ipv6_mapped(),
    ///            Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc000, 0x2ff));
    /// ```
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub const fn to_ipv6_mapped(&self) -> Ipv6Addr {
        let [a, b, c, d] = self.octets();
        Ipv6Addr { octets: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, a, b, c, d] }
    }
}

#[stable(feature = "ip_addr", since = "1.7.0")]
impl fmt::Display for IpAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IpAddr::V4(ip) => ip.fmt(fmt),
            IpAddr::V6(ip) => ip.fmt(fmt),
        }
    }
}

#[stable(feature = "ip_addr", since = "1.7.0")]
impl fmt::Debug for IpAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

#[stable(feature = "ip_from_ip", since = "1.16.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const From<Ipv4Addr> for IpAddr {
    /// Copies this address to a new `IpAddr::V4`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr};
    ///
    /// let addr = Ipv4Addr::new(127, 0, 0, 1);
    ///
    /// assert_eq!(
    ///     IpAddr::V4(addr),
    ///     IpAddr::from(addr)
    /// )
    /// ```
    #[inline]
    fn from(ipv4: Ipv4Addr) -> IpAddr {
        IpAddr::V4(ipv4)
    }
}

#[stable(feature = "ip_from_ip", since = "1.16.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const From<Ipv6Addr> for IpAddr {
    /// Copies this address to a new `IpAddr::V6`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv6Addr};
    ///
    /// let addr = Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff);
    ///
    /// assert_eq!(
    ///     IpAddr::V6(addr),
    ///     IpAddr::from(addr)
    /// );
    /// ```
    #[inline]
    fn from(ipv6: Ipv6Addr) -> IpAddr {
        IpAddr::V6(ipv6)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for Ipv4Addr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let octets = self.octets();

        // If there are no alignment requirements, write the IP address directly to `f`.
        // Otherwise, write it to a local buffer and then use `f.pad`.
        if fmt.precision().is_none() && fmt.width().is_none() {
            write!(fmt, "{}.{}.{}.{}", octets[0], octets[1], octets[2], octets[3])
        } else {
            const LONGEST_IPV4_ADDR: &str = "255.255.255.255";

            let mut buf = DisplayBuffer::<{ LONGEST_IPV4_ADDR.len() }>::new();
            // Buffer is long enough for the longest possible IPv4 address, so this should never fail.
            write!(buf, "{}.{}.{}.{}", octets[0], octets[1], octets[2], octets[3]).unwrap();

            fmt.pad(buf.as_str())
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Ipv4Addr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

#[stable(feature = "ip_cmp", since = "1.16.0")]
impl PartialEq<Ipv4Addr> for IpAddr {
    #[inline]
    fn eq(&self, other: &Ipv4Addr) -> bool {
        match self {
            IpAddr::V4(v4) => v4 == other,
            IpAddr::V6(_) => false,
        }
    }
}

#[stable(feature = "ip_cmp", since = "1.16.0")]
impl PartialEq<IpAddr> for Ipv4Addr {
    #[inline]
    fn eq(&self, other: &IpAddr) -> bool {
        match other {
            IpAddr::V4(v4) => self == v4,
            IpAddr::V6(_) => false,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for Ipv4Addr {
    #[inline]
    fn partial_cmp(&self, other: &Ipv4Addr) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[stable(feature = "ip_cmp", since = "1.16.0")]
impl PartialOrd<Ipv4Addr> for IpAddr {
    #[inline]
    fn partial_cmp(&self, other: &Ipv4Addr) -> Option<Ordering> {
        match self {
            IpAddr::V4(v4) => v4.partial_cmp(other),
            IpAddr::V6(_) => Some(Ordering::Greater),
        }
    }
}

#[stable(feature = "ip_cmp", since = "1.16.0")]
impl PartialOrd<IpAddr> for Ipv4Addr {
    #[inline]
    fn partial_cmp(&self, other: &IpAddr) -> Option<Ordering> {
        match other {
            IpAddr::V4(v4) => self.partial_cmp(v4),
            IpAddr::V6(_) => Some(Ordering::Less),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for Ipv4Addr {
    #[inline]
    fn cmp(&self, other: &Ipv4Addr) -> Ordering {
        self.octets.cmp(&other.octets)
    }
}

#[stable(feature = "ip_u32", since = "1.1.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const From<Ipv4Addr> for u32 {
    /// Uses [`Ipv4Addr::to_bits`] to convert an IPv4 address to a host byte order `u32`.
    #[inline]
    fn from(ip: Ipv4Addr) -> u32 {
        ip.to_bits()
    }
}

#[stable(feature = "ip_u32", since = "1.1.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const From<u32> for Ipv4Addr {
    /// Uses [`Ipv4Addr::from_bits`] to convert a host byte order `u32` into an IPv4 address.
    #[inline]
    fn from(ip: u32) -> Ipv4Addr {
        Ipv4Addr::from_bits(ip)
    }
}

#[stable(feature = "from_slice_v4", since = "1.9.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const From<[u8; 4]> for Ipv4Addr {
    /// Creates an `Ipv4Addr` from a four element byte array.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv4Addr;
    ///
    /// let addr = Ipv4Addr::from([13u8, 12u8, 11u8, 10u8]);
    /// assert_eq!(Ipv4Addr::new(13, 12, 11, 10), addr);
    /// ```
    #[inline]
    fn from(octets: [u8; 4]) -> Ipv4Addr {
        Ipv4Addr { octets }
    }
}

#[stable(feature = "ip_from_slice", since = "1.17.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const From<[u8; 4]> for IpAddr {
    /// Creates an `IpAddr::V4` from a four element byte array.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr};
    ///
    /// let addr = IpAddr::from([13u8, 12u8, 11u8, 10u8]);
    /// assert_eq!(IpAddr::V4(Ipv4Addr::new(13, 12, 11, 10)), addr);
    /// ```
    #[inline]
    fn from(octets: [u8; 4]) -> IpAddr {
        IpAddr::V4(Ipv4Addr::from(octets))
    }
}

impl Ipv6Addr {
    /// Creates a new IPv6 address from eight 16-bit segments.
    ///
    /// The result will represent the IP address `a:b:c:d:e:f:g:h`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// let addr = Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff);
    /// ```
    #[rustc_const_stable(feature = "const_ip_32", since = "1.32.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    pub const fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16, h: u16) -> Ipv6Addr {
        let addr16 = [
            a.to_be(),
            b.to_be(),
            c.to_be(),
            d.to_be(),
            e.to_be(),
            f.to_be(),
            g.to_be(),
            h.to_be(),
        ];
        Ipv6Addr {
            // All elements in `addr16` are big endian.
            // SAFETY: `[u16; 8]` is always safe to transmute to `[u8; 16]`.
            octets: unsafe { transmute::<_, [u8; 16]>(addr16) },
        }
    }

    /// The size of an IPv6 address in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::BITS, 128);
    /// ```
    #[stable(feature = "ip_bits", since = "1.80.0")]
    pub const BITS: u32 = 128;

    /// Converts an IPv6 address into a `u128` representation using native byte order.
    ///
    /// Although IPv6 addresses are big-endian, the `u128` value will use the target platform's
    /// native byte order. That is, the `u128` value is an integer representation of the IPv6
    /// address and not an integer interpretation of the IPv6 address's big-endian bitstring. This
    /// means that the `u128` value masked with `0xffffffffffffffffffffffffffff0000_u128` will set
    /// the last segment in the address to 0, regardless of the target platform's endianness.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// let addr = Ipv6Addr::new(
    ///     0x1020, 0x3040, 0x5060, 0x7080,
    ///     0x90A0, 0xB0C0, 0xD0E0, 0xF00D,
    /// );
    /// assert_eq!(0x102030405060708090A0B0C0D0E0F00D_u128, addr.to_bits());
    /// ```
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// let addr = Ipv6Addr::new(
    ///     0x1020, 0x3040, 0x5060, 0x7080,
    ///     0x90A0, 0xB0C0, 0xD0E0, 0xF00D,
    /// );
    /// let addr_bits = addr.to_bits() & 0xffffffffffffffffffffffffffff0000_u128;
    /// assert_eq!(
    ///     Ipv6Addr::new(
    ///         0x1020, 0x3040, 0x5060, 0x7080,
    ///         0x90A0, 0xB0C0, 0xD0E0, 0x0000,
    ///     ),
    ///     Ipv6Addr::from_bits(addr_bits));
    ///
    /// ```
    #[rustc_const_stable(feature = "ip_bits", since = "1.80.0")]
    #[stable(feature = "ip_bits", since = "1.80.0")]
    #[must_use]
    #[inline]
    pub const fn to_bits(self) -> u128 {
        u128::from_be_bytes(self.octets)
    }

    /// Converts a native byte order `u128` into an IPv6 address.
    ///
    /// See [`Ipv6Addr::to_bits`] for an explanation on endianness.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// let addr = Ipv6Addr::from_bits(0x102030405060708090A0B0C0D0E0F00D_u128);
    /// assert_eq!(
    ///     Ipv6Addr::new(
    ///         0x1020, 0x3040, 0x5060, 0x7080,
    ///         0x90A0, 0xB0C0, 0xD0E0, 0xF00D,
    ///     ),
    ///     addr);
    /// ```
    #[rustc_const_stable(feature = "ip_bits", since = "1.80.0")]
    #[stable(feature = "ip_bits", since = "1.80.0")]
    #[must_use]
    #[inline]
    pub const fn from_bits(bits: u128) -> Ipv6Addr {
        Ipv6Addr { octets: bits.to_be_bytes() }
    }

    /// An IPv6 address representing localhost: `::1`.
    ///
    /// This corresponds to constant `IN6ADDR_LOOPBACK_INIT` or `in6addr_loopback` in other
    /// languages.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// let addr = Ipv6Addr::LOCALHOST;
    /// assert_eq!(addr, Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1));
    /// ```
    #[doc(alias = "IN6ADDR_LOOPBACK_INIT")]
    #[doc(alias = "in6addr_loopback")]
    #[stable(feature = "ip_constructors", since = "1.30.0")]
    pub const LOCALHOST: Self = Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1);

    /// An IPv6 address representing the unspecified address: `::`.
    ///
    /// This corresponds to constant `IN6ADDR_ANY_INIT` or `in6addr_any` in other languages.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// let addr = Ipv6Addr::UNSPECIFIED;
    /// assert_eq!(addr, Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0));
    /// ```
    #[doc(alias = "IN6ADDR_ANY_INIT")]
    #[doc(alias = "in6addr_any")]
    #[stable(feature = "ip_constructors", since = "1.30.0")]
    pub const UNSPECIFIED: Self = Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0);

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
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    pub const fn segments(&self) -> [u16; 8] {
        // All elements in `self.octets` must be big endian.
        // SAFETY: `[u8; 16]` is always safe to transmute to `[u16; 8]`.
        let [a, b, c, d, e, f, g, h] = unsafe { transmute::<_, [u16; 8]>(self.octets) };
        // We want native endian u16
        [
            u16::from_be(a),
            u16::from_be(b),
            u16::from_be(c),
            u16::from_be(d),
            u16::from_be(e),
            u16::from_be(f),
            u16::from_be(g),
            u16::from_be(h),
        ]
    }

    /// Creates an `Ipv6Addr` from an eight element 16-bit array.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// let addr = Ipv6Addr::from_segments([
    ///     0x20du16, 0x20cu16, 0x20bu16, 0x20au16,
    ///     0x209u16, 0x208u16, 0x207u16, 0x206u16,
    /// ]);
    /// assert_eq!(
    ///     Ipv6Addr::new(
    ///         0x20d, 0x20c, 0x20b, 0x20a,
    ///         0x209, 0x208, 0x207, 0x206,
    ///     ),
    ///     addr
    /// );
    /// ```
    #[stable(feature = "ip_from", since = "1.91.0")]
    #[rustc_const_stable(feature = "ip_from", since = "1.91.0")]
    #[must_use]
    #[inline]
    pub const fn from_segments(segments: [u16; 8]) -> Ipv6Addr {
        let [a, b, c, d, e, f, g, h] = segments;
        Ipv6Addr::new(a, b, c, d, e, f, g, h)
    }

    /// Returns [`true`] for the special 'unspecified' address (`::`).
    ///
    /// This property is defined in [IETF RFC 4291].
    ///
    /// [IETF RFC 4291]: https://tools.ietf.org/html/rfc4291
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_unspecified(), false);
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0).is_unspecified(), true);
    /// ```
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(since = "1.7.0", feature = "ip_17")]
    #[must_use]
    #[inline]
    pub const fn is_unspecified(&self) -> bool {
        u128::from_be_bytes(self.octets()) == u128::from_be_bytes(Ipv6Addr::UNSPECIFIED.octets())
    }

    /// Returns [`true`] if this is the [loopback address] (`::1`),
    /// as defined in [IETF RFC 4291 section 2.5.3].
    ///
    /// Contrary to IPv4, in IPv6 there is only one loopback address.
    ///
    /// [loopback address]: Ipv6Addr::LOCALHOST
    /// [IETF RFC 4291 section 2.5.3]: https://tools.ietf.org/html/rfc4291#section-2.5.3
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_loopback(), false);
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0x1).is_loopback(), true);
    /// ```
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(since = "1.7.0", feature = "ip_17")]
    #[must_use]
    #[inline]
    pub const fn is_loopback(&self) -> bool {
        u128::from_be_bytes(self.octets()) == u128::from_be_bytes(Ipv6Addr::LOCALHOST.octets())
    }

    /// Returns [`true`] if the address appears to be globally reachable
    /// as specified by the [IANA IPv6 Special-Purpose Address Registry].
    ///
    /// Whether or not an address is practically reachable will depend on your
    /// network configuration. Most IPv6 addresses are globally reachable, unless
    /// they are specifically defined as *not* globally reachable.
    ///
    /// Non-exhaustive list of notable addresses that are not globally reachable:
    /// - The [unspecified address] ([`is_unspecified`](Ipv6Addr::is_unspecified))
    /// - The [loopback address] ([`is_loopback`](Ipv6Addr::is_loopback))
    /// - IPv4-mapped addresses
    /// - Addresses reserved for benchmarking ([`is_benchmarking`](Ipv6Addr::is_benchmarking))
    /// - Addresses reserved for documentation ([`is_documentation`](Ipv6Addr::is_documentation))
    /// - Unique local addresses ([`is_unique_local`](Ipv6Addr::is_unique_local))
    /// - Unicast addresses with link-local scope ([`is_unicast_link_local`](Ipv6Addr::is_unicast_link_local))
    ///
    /// For the complete overview of which addresses are globally reachable, see the table at the [IANA IPv6 Special-Purpose Address Registry].
    ///
    /// Note that an address having global scope is not the same as being globally reachable,
    /// and there is no direct relation between the two concepts: There exist addresses with global scope
    /// that are not globally reachable (for example unique local addresses),
    /// and addresses that are globally reachable without having global scope
    /// (multicast addresses with non-global scope).
    ///
    /// [IANA IPv6 Special-Purpose Address Registry]: https://www.iana.org/assignments/iana-ipv6-special-registry/iana-ipv6-special-registry.xhtml
    /// [unspecified address]: Ipv6Addr::UNSPECIFIED
    /// [loopback address]: Ipv6Addr::LOCALHOST
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// // Most IPv6 addresses are globally reachable:
    /// assert_eq!(Ipv6Addr::new(0x26, 0, 0x1c9, 0, 0, 0xafc8, 0x10, 0x1).is_global(), true);
    ///
    /// // However some addresses have been assigned a special meaning
    /// // that makes them not globally reachable. Some examples are:
    ///
    /// // The unspecified address (`::`)
    /// assert_eq!(Ipv6Addr::UNSPECIFIED.is_global(), false);
    ///
    /// // The loopback address (`::1`)
    /// assert_eq!(Ipv6Addr::LOCALHOST.is_global(), false);
    ///
    /// // IPv4-mapped addresses (`::ffff:0:0/96`)
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_global(), false);
    ///
    /// // Addresses reserved for benchmarking (`2001:2::/48`)
    /// assert_eq!(Ipv6Addr::new(0x2001, 2, 0, 0, 0, 0, 0, 1,).is_global(), false);
    ///
    /// // Addresses reserved for documentation (`2001:db8::/32` and `3fff::/20`)
    /// assert_eq!(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1).is_global(), false);
    /// assert_eq!(Ipv6Addr::new(0x3fff, 0, 0, 0, 0, 0, 0, 0).is_global(), false);
    ///
    /// // Unique local addresses (`fc00::/7`)
    /// assert_eq!(Ipv6Addr::new(0xfc02, 0, 0, 0, 0, 0, 0, 1).is_global(), false);
    ///
    /// // Unicast addresses with link-local scope (`fe80::/10`)
    /// assert_eq!(Ipv6Addr::new(0xfe81, 0, 0, 0, 0, 0, 0, 1).is_global(), false);
    ///
    /// // For a complete overview see the IANA IPv6 Special-Purpose Address Registry.
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_global(&self) -> bool {
        !(self.is_unspecified()
            || self.is_loopback()
            // IPv4-mapped Address (`::ffff:0:0/96`)
            || matches!(self.segments(), [0, 0, 0, 0, 0, 0xffff, _, _])
            // IPv4-IPv6 Translat. (`64:ff9b:1::/48`)
            || matches!(self.segments(), [0x64, 0xff9b, 1, _, _, _, _, _])
            // Discard-Only Address Block (`100::/64`)
            || matches!(self.segments(), [0x100, 0, 0, 0, _, _, _, _])
            // IETF Protocol Assignments (`2001::/23`)
            || (matches!(self.segments(), [0x2001, b, _, _, _, _, _, _] if b < 0x200)
                && !(
                    // Port Control Protocol Anycast (`2001:1::1`)
                    u128::from_be_bytes(self.octets()) == 0x2001_0001_0000_0000_0000_0000_0000_0001
                    // Traversal Using Relays around NAT Anycast (`2001:1::2`)
                    || u128::from_be_bytes(self.octets()) == 0x2001_0001_0000_0000_0000_0000_0000_0002
                    // AMT (`2001:3::/32`)
                    || matches!(self.segments(), [0x2001, 3, _, _, _, _, _, _])
                    // AS112-v6 (`2001:4:112::/48`)
                    || matches!(self.segments(), [0x2001, 4, 0x112, _, _, _, _, _])
                    // ORCHIDv2 (`2001:20::/28`)
                    // Drone Remote ID Protocol Entity Tags (DETs) Prefix (`2001:30::/28`)`
                    || matches!(self.segments(), [0x2001, b, _, _, _, _, _, _] if b >= 0x20 && b <= 0x3F)
                ))
            // 6to4 (`2002::/16`) – it's not explicitly documented as globally reachable,
            // IANA says N/A.
            || matches!(self.segments(), [0x2002, _, _, _, _, _, _, _])
            || self.is_documentation()
            // Segment Routing (SRv6) SIDs (`5f00::/16`)
            || matches!(self.segments(), [0x5f00, ..])
            || self.is_unique_local()
            || self.is_unicast_link_local())
    }

    /// Returns [`true`] if this is a unique local address (`fc00::/7`).
    ///
    /// This property is defined in [IETF RFC 4193].
    ///
    /// [IETF RFC 4193]: https://tools.ietf.org/html/rfc4193
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_unique_local(), false);
    /// assert_eq!(Ipv6Addr::new(0xfc02, 0, 0, 0, 0, 0, 0, 0).is_unique_local(), true);
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "ipv6_is_unique_local", since = "1.84.0")]
    #[rustc_const_stable(feature = "ipv6_is_unique_local", since = "1.84.0")]
    pub const fn is_unique_local(&self) -> bool {
        (self.segments()[0] & 0xfe00) == 0xfc00
    }

    /// Returns [`true`] if this is a unicast address, as defined by [IETF RFC 4291].
    /// Any address that is not a [multicast address] (`ff00::/8`) is unicast.
    ///
    /// [IETF RFC 4291]: https://tools.ietf.org/html/rfc4291
    /// [multicast address]: Ipv6Addr::is_multicast
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// // The unspecified and loopback addresses are unicast.
    /// assert_eq!(Ipv6Addr::UNSPECIFIED.is_unicast(), true);
    /// assert_eq!(Ipv6Addr::LOCALHOST.is_unicast(), true);
    ///
    /// // Any address that is not a multicast address (`ff00::/8`) is unicast.
    /// assert_eq!(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0).is_unicast(), true);
    /// assert_eq!(Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0).is_unicast(), false);
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_unicast(&self) -> bool {
        !self.is_multicast()
    }

    /// Returns `true` if the address is a unicast address with link-local scope,
    /// as defined in [RFC 4291].
    ///
    /// A unicast address has link-local scope if it has the prefix `fe80::/10`, as per [RFC 4291 section 2.4].
    /// Note that this encompasses more addresses than those defined in [RFC 4291 section 2.5.6],
    /// which describes "Link-Local IPv6 Unicast Addresses" as having the following stricter format:
    ///
    /// ```text
    /// | 10 bits  |         54 bits         |          64 bits           |
    /// +----------+-------------------------+----------------------------+
    /// |1111111010|           0             |       interface ID         |
    /// +----------+-------------------------+----------------------------+
    /// ```
    /// So while currently the only addresses with link-local scope an application will encounter are all in `fe80::/64`,
    /// this might change in the future with the publication of new standards. More addresses in `fe80::/10` could be allocated,
    /// and those addresses will have link-local scope.
    ///
    /// Also note that while [RFC 4291 section 2.5.3] mentions about the [loopback address] (`::1`) that "it is treated as having Link-Local scope",
    /// this does not mean that the loopback address actually has link-local scope and this method will return `false` on it.
    ///
    /// [RFC 4291]: https://tools.ietf.org/html/rfc4291
    /// [RFC 4291 section 2.4]: https://tools.ietf.org/html/rfc4291#section-2.4
    /// [RFC 4291 section 2.5.3]: https://tools.ietf.org/html/rfc4291#section-2.5.3
    /// [RFC 4291 section 2.5.6]: https://tools.ietf.org/html/rfc4291#section-2.5.6
    /// [loopback address]: Ipv6Addr::LOCALHOST
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// // The loopback address (`::1`) does not actually have link-local scope.
    /// assert_eq!(Ipv6Addr::LOCALHOST.is_unicast_link_local(), false);
    ///
    /// // Only addresses in `fe80::/10` have link-local scope.
    /// assert_eq!(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0).is_unicast_link_local(), false);
    /// assert_eq!(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 0).is_unicast_link_local(), true);
    ///
    /// // Addresses outside the stricter `fe80::/64` also have link-local scope.
    /// assert_eq!(Ipv6Addr::new(0xfe80, 0, 0, 1, 0, 0, 0, 0).is_unicast_link_local(), true);
    /// assert_eq!(Ipv6Addr::new(0xfe81, 0, 0, 0, 0, 0, 0, 0).is_unicast_link_local(), true);
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "ipv6_is_unique_local", since = "1.84.0")]
    #[rustc_const_stable(feature = "ipv6_is_unique_local", since = "1.84.0")]
    pub const fn is_unicast_link_local(&self) -> bool {
        (self.segments()[0] & 0xffc0) == 0xfe80
    }

    /// Returns [`true`] if this is an address reserved for documentation
    /// (`2001:db8::/32` and `3fff::/20`).
    ///
    /// This property is defined by [IETF RFC 3849] and [IETF RFC 9637].
    ///
    /// [IETF RFC 3849]: https://tools.ietf.org/html/rfc3849
    /// [IETF RFC 9637]: https://tools.ietf.org/html/rfc9637
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_documentation(), false);
    /// assert_eq!(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0).is_documentation(), true);
    /// assert_eq!(Ipv6Addr::new(0x3fff, 0, 0, 0, 0, 0, 0, 0).is_documentation(), true);
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_documentation(&self) -> bool {
        matches!(self.segments(), [0x2001, 0xdb8, ..] | [0x3fff, 0..=0x0fff, ..])
    }

    /// Returns [`true`] if this is an address reserved for benchmarking (`2001:2::/48`).
    ///
    /// This property is defined in [IETF RFC 5180], where it is mistakenly specified as covering the range `2001:0200::/48`.
    /// This is corrected in [IETF RFC Errata 1752] to `2001:0002::/48`.
    ///
    /// [IETF RFC 5180]: https://tools.ietf.org/html/rfc5180
    /// [IETF RFC Errata 1752]: https://www.rfc-editor.org/errata_search.php?eid=1752
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc613, 0x0).is_benchmarking(), false);
    /// assert_eq!(Ipv6Addr::new(0x2001, 0x2, 0, 0, 0, 0, 0, 0).is_benchmarking(), true);
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_benchmarking(&self) -> bool {
        (self.segments()[0] == 0x2001) && (self.segments()[1] == 0x2) && (self.segments()[2] == 0)
    }

    /// Returns [`true`] if the address is a globally routable unicast address.
    ///
    /// The following return false:
    ///
    /// - the loopback address
    /// - the link-local addresses
    /// - unique local addresses
    /// - the unspecified address
    /// - the address range reserved for documentation
    ///
    /// This method returns [`true`] for site-local addresses as per [RFC 4291 section 2.5.7]
    ///
    /// ```no_rust
    /// The special behavior of [the site-local unicast] prefix defined in [RFC3513] must no longer
    /// be supported in new implementations (i.e., new implementations must treat this prefix as
    /// Global Unicast).
    /// ```
    ///
    /// [RFC 4291 section 2.5.7]: https://tools.ietf.org/html/rfc4291#section-2.5.7
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0).is_unicast_global(), false);
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_unicast_global(), true);
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_unicast_global(&self) -> bool {
        self.is_unicast()
            && !self.is_loopback()
            && !self.is_unicast_link_local()
            && !self.is_unique_local()
            && !self.is_unspecified()
            && !self.is_documentation()
            && !self.is_benchmarking()
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
    /// assert_eq!(
    ///     Ipv6Addr::new(0xff0e, 0, 0, 0, 0, 0, 0, 0).multicast_scope(),
    ///     Some(Ipv6MulticastScope::Global)
    /// );
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).multicast_scope(), None);
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn multicast_scope(&self) -> Option<Ipv6MulticastScope> {
        if self.is_multicast() {
            match self.segments()[0] & 0x000f {
                1 => Some(Ipv6MulticastScope::InterfaceLocal),
                2 => Some(Ipv6MulticastScope::LinkLocal),
                3 => Some(Ipv6MulticastScope::RealmLocal),
                4 => Some(Ipv6MulticastScope::AdminLocal),
                5 => Some(Ipv6MulticastScope::SiteLocal),
                8 => Some(Ipv6MulticastScope::OrganizationLocal),
                14 => Some(Ipv6MulticastScope::Global),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Returns [`true`] if this is a multicast address (`ff00::/8`).
    ///
    /// This property is defined by [IETF RFC 4291].
    ///
    /// [IETF RFC 4291]: https://tools.ietf.org/html/rfc4291
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0).is_multicast(), true);
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).is_multicast(), false);
    /// ```
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(since = "1.7.0", feature = "ip_17")]
    #[must_use]
    #[inline]
    pub const fn is_multicast(&self) -> bool {
        (self.segments()[0] & 0xff00) == 0xff00
    }

    /// Returns [`true`] if the address is an IPv4-mapped address (`::ffff:0:0/96`).
    ///
    /// IPv4-mapped addresses can be converted to their canonical IPv4 address with
    /// [`to_ipv4_mapped`](Ipv6Addr::to_ipv4_mapped).
    ///
    /// # Examples
    /// ```
    /// #![feature(ip)]
    ///
    /// use std::net::{Ipv4Addr, Ipv6Addr};
    ///
    /// let ipv4_mapped = Ipv4Addr::new(192, 0, 2, 255).to_ipv6_mapped();
    /// assert_eq!(ipv4_mapped.is_ipv4_mapped(), true);
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc000, 0x2ff).is_ipv4_mapped(), true);
    ///
    /// assert_eq!(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0).is_ipv4_mapped(), false);
    /// ```
    #[unstable(feature = "ip", issue = "27709")]
    #[must_use]
    #[inline]
    pub const fn is_ipv4_mapped(&self) -> bool {
        matches!(self.segments(), [0, 0, 0, 0, 0, 0xffff, _, _])
    }

    /// Converts this address to an [`IPv4` address] if it's an [IPv4-mapped] address,
    /// as defined in [IETF RFC 4291 section 2.5.5.2], otherwise returns [`None`].
    ///
    /// `::ffff:a.b.c.d` becomes `a.b.c.d`.
    /// All addresses *not* starting with `::ffff` will return `None`.
    ///
    /// [`IPv4` address]: Ipv4Addr
    /// [IPv4-mapped]: Ipv6Addr
    /// [IETF RFC 4291 section 2.5.5.2]: https://tools.ietf.org/html/rfc4291#section-2.5.5.2
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{Ipv4Addr, Ipv6Addr};
    ///
    /// assert_eq!(Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0).to_ipv4_mapped(), None);
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0xc00a, 0x2ff).to_ipv4_mapped(),
    ///            Some(Ipv4Addr::new(192, 10, 2, 255)));
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1).to_ipv4_mapped(), None);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "ipv6_to_ipv4_mapped", since = "1.63.0")]
    #[rustc_const_stable(feature = "const_ipv6_to_ipv4_mapped", since = "1.75.0")]
    pub const fn to_ipv4_mapped(&self) -> Option<Ipv4Addr> {
        match self.octets() {
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, a, b, c, d] => {
                Some(Ipv4Addr::new(a, b, c, d))
            }
            _ => None,
        }
    }

    /// Converts this address to an [`IPv4` address] if it is either
    /// an [IPv4-compatible] address as defined in [IETF RFC 4291 section 2.5.5.1],
    /// or an [IPv4-mapped] address as defined in [IETF RFC 4291 section 2.5.5.2],
    /// otherwise returns [`None`].
    ///
    /// Note that this will return an [`IPv4` address] for the IPv6 loopback address `::1`. Use
    /// [`Ipv6Addr::to_ipv4_mapped`] to avoid this.
    ///
    /// `::a.b.c.d` and `::ffff:a.b.c.d` become `a.b.c.d`. `::1` becomes `0.0.0.1`.
    /// All addresses *not* starting with either all zeroes or `::ffff` will return `None`.
    ///
    /// [`IPv4` address]: Ipv4Addr
    /// [IPv4-compatible]: Ipv6Addr#ipv4-compatible-ipv6-addresses
    /// [IPv4-mapped]: Ipv6Addr#ipv4-mapped-ipv6-addresses
    /// [IETF RFC 4291 section 2.5.5.1]: https://tools.ietf.org/html/rfc4291#section-2.5.5.1
    /// [IETF RFC 4291 section 2.5.5.2]: https://tools.ietf.org/html/rfc4291#section-2.5.5.2
    ///
    /// # Examples
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
    #[rustc_const_stable(feature = "const_ip_50", since = "1.50.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub const fn to_ipv4(&self) -> Option<Ipv4Addr> {
        if let [0, 0, 0, 0, 0, 0 | 0xffff, ab, cd] = self.segments() {
            let [a, b] = ab.to_be_bytes();
            let [c, d] = cd.to_be_bytes();
            Some(Ipv4Addr::new(a, b, c, d))
        } else {
            None
        }
    }

    /// Converts this address to an `IpAddr::V4` if it is an IPv4-mapped address,
    /// otherwise returns self wrapped in an `IpAddr::V6`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x7f00, 0x1).is_loopback(), false);
    /// assert_eq!(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x7f00, 0x1).to_canonical().is_loopback(), true);
    /// ```
    #[inline]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[stable(feature = "ip_to_canonical", since = "1.75.0")]
    #[rustc_const_stable(feature = "ip_to_canonical", since = "1.75.0")]
    pub const fn to_canonical(&self) -> IpAddr {
        if let Some(mapped) = self.to_ipv4_mapped() {
            return IpAddr::V4(mapped);
        }
        IpAddr::V6(*self)
    }

    /// Returns the sixteen eight-bit integers the IPv6 address consists of.
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0).octets(),
    ///            [0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    /// ```
    #[rustc_const_stable(feature = "const_ip_32", since = "1.32.0")]
    #[stable(feature = "ipv6_to_octets", since = "1.12.0")]
    #[must_use]
    #[inline]
    pub const fn octets(&self) -> [u8; 16] {
        self.octets
    }

    /// Creates an `Ipv6Addr` from a sixteen element byte array.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// let addr = Ipv6Addr::from_octets([
    ///     0x19u8, 0x18u8, 0x17u8, 0x16u8, 0x15u8, 0x14u8, 0x13u8, 0x12u8,
    ///     0x11u8, 0x10u8, 0x0fu8, 0x0eu8, 0x0du8, 0x0cu8, 0x0bu8, 0x0au8,
    /// ]);
    /// assert_eq!(
    ///     Ipv6Addr::new(
    ///         0x1918, 0x1716, 0x1514, 0x1312,
    ///         0x1110, 0x0f0e, 0x0d0c, 0x0b0a,
    ///     ),
    ///     addr
    /// );
    /// ```
    #[stable(feature = "ip_from", since = "1.91.0")]
    #[rustc_const_stable(feature = "ip_from", since = "1.91.0")]
    #[must_use]
    #[inline]
    pub const fn from_octets(octets: [u8; 16]) -> Ipv6Addr {
        Ipv6Addr { octets }
    }

    /// Returns the sixteen eight-bit integers the IPv6 address consists of
    /// as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ip_as_octets)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// assert_eq!(Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0).as_octets(),
    ///            &[255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    /// ```
    #[unstable(feature = "ip_as_octets", issue = "137259")]
    #[inline]
    pub const fn as_octets(&self) -> &[u8; 16] {
        &self.octets
    }
}

/// Writes an Ipv6Addr, conforming to the canonical style described by
/// [RFC 5952](https://tools.ietf.org/html/rfc5952).
#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for Ipv6Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // If there are no alignment requirements, write the IP address directly to `f`.
        // Otherwise, write it to a local buffer and then use `f.pad`.
        if f.precision().is_none() && f.width().is_none() {
            let segments = self.segments();

            if let Some(ipv4) = self.to_ipv4_mapped() {
                write!(f, "::ffff:{}", ipv4)
            } else {
                #[derive(Copy, Clone, Default)]
                struct Span {
                    start: usize,
                    len: usize,
                }

                // Find the inner 0 span
                let zeroes = {
                    let mut longest = Span::default();
                    let mut current = Span::default();

                    for (i, &segment) in segments.iter().enumerate() {
                        if segment == 0 {
                            if current.len == 0 {
                                current.start = i;
                            }

                            current.len += 1;

                            if current.len > longest.len {
                                longest = current;
                            }
                        } else {
                            current = Span::default();
                        }
                    }

                    longest
                };

                /// Writes a colon-separated part of the address.
                #[inline]
                fn fmt_subslice(f: &mut fmt::Formatter<'_>, chunk: &[u16]) -> fmt::Result {
                    if let Some((first, tail)) = chunk.split_first() {
                        write!(f, "{:x}", first)?;
                        for segment in tail {
                            f.write_char(':')?;
                            write!(f, "{:x}", segment)?;
                        }
                    }
                    Ok(())
                }

                if zeroes.len > 1 {
                    fmt_subslice(f, &segments[..zeroes.start])?;
                    f.write_str("::")?;
                    fmt_subslice(f, &segments[zeroes.start + zeroes.len..])
                } else {
                    fmt_subslice(f, &segments)
                }
            }
        } else {
            const LONGEST_IPV6_ADDR: &str = "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff";

            let mut buf = DisplayBuffer::<{ LONGEST_IPV6_ADDR.len() }>::new();
            // Buffer is long enough for the longest possible IPv6 address, so this should never fail.
            write!(buf, "{}", self).unwrap();

            f.pad(buf.as_str())
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Ipv6Addr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

#[stable(feature = "ip_cmp", since = "1.16.0")]
impl PartialEq<IpAddr> for Ipv6Addr {
    #[inline]
    fn eq(&self, other: &IpAddr) -> bool {
        match other {
            IpAddr::V4(_) => false,
            IpAddr::V6(v6) => self == v6,
        }
    }
}

#[stable(feature = "ip_cmp", since = "1.16.0")]
impl PartialEq<Ipv6Addr> for IpAddr {
    #[inline]
    fn eq(&self, other: &Ipv6Addr) -> bool {
        match self {
            IpAddr::V4(_) => false,
            IpAddr::V6(v6) => v6 == other,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for Ipv6Addr {
    #[inline]
    fn partial_cmp(&self, other: &Ipv6Addr) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[stable(feature = "ip_cmp", since = "1.16.0")]
impl PartialOrd<Ipv6Addr> for IpAddr {
    #[inline]
    fn partial_cmp(&self, other: &Ipv6Addr) -> Option<Ordering> {
        match self {
            IpAddr::V4(_) => Some(Ordering::Less),
            IpAddr::V6(v6) => v6.partial_cmp(other),
        }
    }
}

#[stable(feature = "ip_cmp", since = "1.16.0")]
impl PartialOrd<IpAddr> for Ipv6Addr {
    #[inline]
    fn partial_cmp(&self, other: &IpAddr) -> Option<Ordering> {
        match other {
            IpAddr::V4(_) => Some(Ordering::Greater),
            IpAddr::V6(v6) => self.partial_cmp(v6),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for Ipv6Addr {
    #[inline]
    fn cmp(&self, other: &Ipv6Addr) -> Ordering {
        self.segments().cmp(&other.segments())
    }
}

#[stable(feature = "i128", since = "1.26.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const From<Ipv6Addr> for u128 {
    /// Uses [`Ipv6Addr::to_bits`] to convert an IPv6 address to a host byte order `u128`.
    #[inline]
    fn from(ip: Ipv6Addr) -> u128 {
        ip.to_bits()
    }
}
#[stable(feature = "i128", since = "1.26.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const From<u128> for Ipv6Addr {
    /// Uses [`Ipv6Addr::from_bits`] to convert a host byte order `u128` to an IPv6 address.
    #[inline]
    fn from(ip: u128) -> Ipv6Addr {
        Ipv6Addr::from_bits(ip)
    }
}

#[stable(feature = "ipv6_from_octets", since = "1.9.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const From<[u8; 16]> for Ipv6Addr {
    /// Creates an `Ipv6Addr` from a sixteen element byte array.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// let addr = Ipv6Addr::from([
    ///     0x19u8, 0x18u8, 0x17u8, 0x16u8, 0x15u8, 0x14u8, 0x13u8, 0x12u8,
    ///     0x11u8, 0x10u8, 0x0fu8, 0x0eu8, 0x0du8, 0x0cu8, 0x0bu8, 0x0au8,
    /// ]);
    /// assert_eq!(
    ///     Ipv6Addr::new(
    ///         0x1918, 0x1716, 0x1514, 0x1312,
    ///         0x1110, 0x0f0e, 0x0d0c, 0x0b0a,
    ///     ),
    ///     addr
    /// );
    /// ```
    #[inline]
    fn from(octets: [u8; 16]) -> Ipv6Addr {
        Ipv6Addr { octets }
    }
}

#[stable(feature = "ipv6_from_segments", since = "1.16.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const From<[u16; 8]> for Ipv6Addr {
    /// Creates an `Ipv6Addr` from an eight element 16-bit array.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::Ipv6Addr;
    ///
    /// let addr = Ipv6Addr::from([
    ///     0x20du16, 0x20cu16, 0x20bu16, 0x20au16,
    ///     0x209u16, 0x208u16, 0x207u16, 0x206u16,
    /// ]);
    /// assert_eq!(
    ///     Ipv6Addr::new(
    ///         0x20d, 0x20c, 0x20b, 0x20a,
    ///         0x209, 0x208, 0x207, 0x206,
    ///     ),
    ///     addr
    /// );
    /// ```
    #[inline]
    fn from(segments: [u16; 8]) -> Ipv6Addr {
        let [a, b, c, d, e, f, g, h] = segments;
        Ipv6Addr::new(a, b, c, d, e, f, g, h)
    }
}

#[stable(feature = "ip_from_slice", since = "1.17.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const From<[u8; 16]> for IpAddr {
    /// Creates an `IpAddr::V6` from a sixteen element byte array.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv6Addr};
    ///
    /// let addr = IpAddr::from([
    ///     0x19u8, 0x18u8, 0x17u8, 0x16u8, 0x15u8, 0x14u8, 0x13u8, 0x12u8,
    ///     0x11u8, 0x10u8, 0x0fu8, 0x0eu8, 0x0du8, 0x0cu8, 0x0bu8, 0x0au8,
    /// ]);
    /// assert_eq!(
    ///     IpAddr::V6(Ipv6Addr::new(
    ///         0x1918, 0x1716, 0x1514, 0x1312,
    ///         0x1110, 0x0f0e, 0x0d0c, 0x0b0a,
    ///     )),
    ///     addr
    /// );
    /// ```
    #[inline]
    fn from(octets: [u8; 16]) -> IpAddr {
        IpAddr::V6(Ipv6Addr::from(octets))
    }
}

#[stable(feature = "ip_from_slice", since = "1.17.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const From<[u16; 8]> for IpAddr {
    /// Creates an `IpAddr::V6` from an eight element 16-bit array.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv6Addr};
    ///
    /// let addr = IpAddr::from([
    ///     0x20du16, 0x20cu16, 0x20bu16, 0x20au16,
    ///     0x209u16, 0x208u16, 0x207u16, 0x206u16,
    /// ]);
    /// assert_eq!(
    ///     IpAddr::V6(Ipv6Addr::new(
    ///         0x20d, 0x20c, 0x20b, 0x20a,
    ///         0x209, 0x208, 0x207, 0x206,
    ///     )),
    ///     addr
    /// );
    /// ```
    #[inline]
    fn from(segments: [u16; 8]) -> IpAddr {
        IpAddr::V6(Ipv6Addr::from(segments))
    }
}

#[stable(feature = "ip_bitops", since = "1.75.0")]
#[rustc_const_unstable(feature = "const_ops", issue = "143802")]
impl const Not for Ipv4Addr {
    type Output = Ipv4Addr;

    #[inline]
    fn not(mut self) -> Ipv4Addr {
        let mut idx = 0;
        while idx < 4 {
            self.octets[idx] = !self.octets[idx];
            idx += 1;
        }
        self
    }
}

#[stable(feature = "ip_bitops", since = "1.75.0")]
#[rustc_const_unstable(feature = "const_ops", issue = "143802")]
impl const Not for &'_ Ipv4Addr {
    type Output = Ipv4Addr;

    #[inline]
    fn not(self) -> Ipv4Addr {
        !*self
    }
}

#[stable(feature = "ip_bitops", since = "1.75.0")]
#[rustc_const_unstable(feature = "const_ops", issue = "143802")]
impl const Not for Ipv6Addr {
    type Output = Ipv6Addr;

    #[inline]
    fn not(mut self) -> Ipv6Addr {
        let mut idx = 0;
        while idx < 16 {
            self.octets[idx] = !self.octets[idx];
            idx += 1;
        }
        self
    }
}

#[stable(feature = "ip_bitops", since = "1.75.0")]
#[rustc_const_unstable(feature = "const_ops", issue = "143802")]
impl const Not for &'_ Ipv6Addr {
    type Output = Ipv6Addr;

    #[inline]
    fn not(self) -> Ipv6Addr {
        !*self
    }
}

macro_rules! bitop_impls {
    ($(
        $(#[$attr:meta])*
        impl ($BitOp:ident, $BitOpAssign:ident) for $ty:ty = ($bitop:ident, $bitop_assign:ident);
    )*) => {
        $(
            $(#[$attr])*
            impl const $BitOpAssign for $ty {
                fn $bitop_assign(&mut self, rhs: $ty) {
                    let mut idx = 0;
                    while idx < self.octets.len() {
                        self.octets[idx].$bitop_assign(rhs.octets[idx]);
                        idx += 1;
                    }
                }
            }

            $(#[$attr])*
            impl const $BitOpAssign<&'_ $ty> for $ty {
                fn $bitop_assign(&mut self, rhs: &'_ $ty) {
                    self.$bitop_assign(*rhs);
                }
            }

            $(#[$attr])*
            impl const $BitOp for $ty {
                type Output = $ty;

                #[inline]
                fn $bitop(mut self, rhs: $ty) -> $ty {
                    self.$bitop_assign(rhs);
                    self
                }
            }

            $(#[$attr])*
            impl const $BitOp<&'_ $ty> for $ty {
                type Output = $ty;

                #[inline]
                fn $bitop(mut self, rhs: &'_ $ty) -> $ty {
                    self.$bitop_assign(*rhs);
                    self
                }
            }

            $(#[$attr])*
            impl const $BitOp<$ty> for &'_ $ty {
                type Output = $ty;

                #[inline]
                fn $bitop(self, rhs: $ty) -> $ty {
                    let mut lhs = *self;
                    lhs.$bitop_assign(rhs);
                    lhs
                }
            }

            $(#[$attr])*
            impl const $BitOp<&'_ $ty> for &'_ $ty {
                type Output = $ty;

                #[inline]
                fn $bitop(self, rhs: &'_ $ty) -> $ty {
                    let mut lhs = *self;
                    lhs.$bitop_assign(*rhs);
                    lhs
                }
            }
        )*
    };
}

bitop_impls! {
    #[stable(feature = "ip_bitops", since = "1.75.0")]
    #[rustc_const_unstable(feature = "const_ops", issue = "143802")]
    impl (BitAnd, BitAndAssign) for Ipv4Addr = (bitand, bitand_assign);
    #[stable(feature = "ip_bitops", since = "1.75.0")]
    #[rustc_const_unstable(feature = "const_ops", issue = "143802")]
    impl (BitOr, BitOrAssign) for Ipv4Addr = (bitor, bitor_assign);

    #[stable(feature = "ip_bitops", since = "1.75.0")]
    #[rustc_const_unstable(feature = "const_ops", issue = "143802")]
    impl (BitAnd, BitAndAssign) for Ipv6Addr = (bitand, bitand_assign);
    #[stable(feature = "ip_bitops", since = "1.75.0")]
    #[rustc_const_unstable(feature = "const_ops", issue = "143802")]
    impl (BitOr, BitOrAssign) for Ipv6Addr = (bitor, bitor_assign);
}
