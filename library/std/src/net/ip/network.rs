use super::{Ipv4Addr, Ipv6Addr};
use crate::error::Error;
use crate::fmt;

/// Represents an [IPv4 address] prefix, describing a network of addresses.
///
/// An IPv4 address prefix can be represented by an IPv4 address and a prefix length.
/// The prefix length specifies how many of the leftmost contiguous bits of the
/// address comprise the prefix. The prefix then describes the network containing all
/// addresses that start with the same bits as the prefix.
///
/// The size of an `Ipv6AddrPrefix` struct may vary depending on the target operating system.
///
/// # Textual representation
///
/// `Ipv4AddrPrefix` provides a [`FromStr`](crate::str::FromStr) implementation.
/// The textual representation of an address prefix is the representation of an
/// IPv4 address, followed by a separating `/` and then the prefix length in decimal notation;
/// for example `192.0.2.0/24`.
///
///
/// # Examples
/// ```
/// #![feature(ip_prefix)]
///
/// use std::net::{Ipv4Addr, Ipv4AddrPrefix};
///
/// // Create the address prefix `192.0.2.0/24`.
/// let prefix = Ipv4AddrPrefix::new(Ipv4Addr::new(192, 0, 2, 0), 24).unwrap();
///
/// // The prefix is uniquely defined by a prefix address and prefix length.
/// assert_eq!(prefix.address(), Ipv4Addr::new(192, 0, 2, 0));
/// assert_eq!(prefix.len(), 24);
///
/// // The prefix describes the network of all addresses that start
/// // with the same bits as the prefix.
/// assert_eq!(prefix.contains(&Ipv4Addr::new(192, 0, 2, 0)), true);
/// assert_eq!(prefix.contains(&Ipv4Addr::new(192, 0, 2, 7)), true);
/// assert_eq!(prefix.contains(&Ipv4Addr::new(192, 0, 2, 255)), true);
/// assert_eq!(prefix.contains(&Ipv4Addr::new(192, 0, 3, 0)), false);
/// ```
///
/// [IPv4 address]: Ipv4Addr
#[derive(Copy, PartialEq, Eq, Clone, Hash)]
#[unstable(feature = "ip_prefix", issue = "86991")]
pub struct Ipv4AddrPrefix {
    address_raw: u32,
    len: u8,
}

/// Represents an [IPv6 address] prefix, as described in [IETF RFC 4291 section 2.3].
///
/// An IPv6 address prefix can be represented by an IPv4 address and a prefix length.
/// The prefix length specifies how many of the leftmost contiguous bits of the
/// address comprise the prefix. The prefix then describes the network containing all
/// addresses that start with the same bits as the prefix.
///
/// The size of an `Ipv6AddrPrefix` struct may vary depending on the target operating system.
///
/// # Textual representation
///
/// `Ipv6AddrPrefix` provides a [`FromStr`](crate::str::FromStr)  implementation.
/// The textual representation of an address prefix is the representation of an
/// IPv6 address, followed by a separating `/` and then the prefix length in decimal notation;
/// for example `2001:db8::/32`.
///
/// # Examples
/// ```
/// #![feature(ip_prefix)]
///
/// use std::net::{Ipv6Addr, Ipv6AddrPrefix};
///
/// // Create the address prefix `2001:db8::/32`.
/// let prefix = Ipv6AddrPrefix::new(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0), 32).unwrap();
///
/// // The prefix is uniquely defined by a prefix address and prefix length.
/// assert_eq!(prefix.address(), Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0));
/// assert_eq!(prefix.len(), 32);
///
/// // The prefix describes the network of all addresses that start
/// // with the same bits as the prefix.
/// assert_eq!(prefix.contains(&Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0)), true);
/// assert_eq!(prefix.contains(&Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 7)), true);
/// assert_eq!(prefix.contains(&Ipv6Addr::new(0x2001, 0xdb8, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff)), true);
/// assert_eq!(prefix.contains(&Ipv6Addr::new(0x2001, 0xdb9, 0, 0, 0, 0, 0, 0)), false);
/// ```
///
/// [IPv6 address]: Ipv6Addr
/// [IETF RFC 4291 section 2.3]: https://tools.ietf.org/html/rfc4291#section-2.3
#[derive(Copy, PartialEq, Eq, Clone, Hash)]
#[unstable(feature = "ip_prefix", issue = "86991")]
pub struct Ipv6AddrPrefix {
    address_raw: u128,
    len: u8,
}

impl Ipv4AddrPrefix {
    /// Creates a new IPv4 address prefix from an address and a prefix length.
    ///
    /// # Examples
    /// ```
    /// #![feature(ip_prefix)]
    ///
    /// use std::net::{Ipv4Addr, Ipv4AddrPrefix};
    ///
    /// // Create the address prefix `192.0.2.0/24`.
    /// let prefix = Ipv4AddrPrefix::new(Ipv4Addr::new(192, 0, 2, 0), 24).unwrap();
    ///
    /// // Error: Prefix length can not be more than 32 bits.
    /// Ipv4AddrPrefix::new(Ipv4Addr::new(192, 0, 2, 0), 35).unwrap_err();
    /// ```
    #[unstable(feature = "ip_prefix", issue = "86991")]
    #[inline]
    pub const fn new(address: Ipv4Addr, len: u32) -> Result<Ipv4AddrPrefix, InvalidPrefixError> {
        if len <= u32::BITS {
            Ok(Ipv4AddrPrefix::new_unchecked(address, len))
        } else {
            Err(InvalidPrefixError { max_len: u32::BITS as u8 })
        }
    }

    // Private constructor that assumes len <= 32.
    // Useful because `Result::unwrap` is not yet usable in const contexts, so `new` can't be used.
    pub(crate) const fn new_unchecked(address: Ipv4Addr, len: u32) -> Ipv4AddrPrefix {
        let masked = {
            let mask = Ipv4AddrPrefix::mask(len);
            u32::from_be_bytes(address.octets()) & mask
        };

        Ipv4AddrPrefix { address_raw: masked, len: len as u8 }
    }

    /// Returns the address specifying this address prefix.
    ///
    /// The prefix address and the prefix length together uniquely define an address prefix.
    ///
    /// # Examples
    /// ```
    /// #![feature(ip_prefix)]
    ///
    /// use std::net::{Ipv4Addr, Ipv4AddrPrefix};
    ///
    /// // `192.0.2.0/24`
    /// let prefix = Ipv4AddrPrefix::new(Ipv4Addr::new(192, 0, 2, 7), 24).unwrap();
    ///
    /// // Note that the address can be different than the address the
    /// // prefix was constructed with. The returned address only contains the bits
    /// // specified by the prefix length.
    /// assert_eq!(prefix.address(), Ipv4Addr::new(192, 0, 2, 0));
    /// assert_ne!(prefix.address(), Ipv4Addr::new(192, 0, 2, 7));
    /// ```
    #[unstable(feature = "ip_prefix", issue = "86991")]
    #[inline]
    pub const fn address(&self) -> Ipv4Addr {
        Ipv4Addr::from_u32(self.address_raw)
    }

    /// Returns the prefix length of this address prefix.
    ///
    /// The prefix address and the prefix length together uniquely define an address prefix.
    ///
    /// # Examples
    /// ```
    /// #![feature(ip_prefix)]
    ///
    /// use std::net::{Ipv4Addr, Ipv4AddrPrefix};
    ///
    /// // `192.0.2.0/24`
    /// let prefix = Ipv4AddrPrefix::new(Ipv4Addr::new(192, 0, 2, 0), 24).unwrap();
    ///
    /// assert_eq!(prefix.len(), 24);
    /// ```
    #[unstable(feature = "ip_prefix", issue = "86991")]
    #[inline]
    pub const fn len(&self) -> u32 {
        self.len as u32
    }

    // Compute the bitmask specified by a prefix length.
    #[inline]
    const fn mask(len: u32) -> u32 {
        if len == 0 {
            0
        } else {
            // shift will not overflow as len > 0, so u32::BITS - len < 32
            u32::MAX << (u32::BITS - len)
        }
    }

    /// Returns `true` if the given address is contained in the network described by this prefix,
    /// meaning that it starts with the same bits as the prefix.
    ///
    /// # Examples
    /// ```
    /// #![feature(ip_prefix)]
    ///
    /// use std::net::{Ipv4Addr, Ipv4AddrPrefix};
    ///
    /// // `192.0.2.0/24`
    /// let prefix = Ipv4AddrPrefix::new(Ipv4Addr::new(192, 0, 2, 0), 24).unwrap();
    ///
    /// assert_eq!(prefix.contains(&Ipv4Addr::new(192, 0, 2, 7)), true);
    /// assert_eq!(prefix.contains(&Ipv4Addr::new(192, 0, 3, 0)), false);
    /// ```
    #[unstable(feature = "ip_prefix", issue = "86991")]
    #[inline]
    pub const fn contains(&self, address: &Ipv4Addr) -> bool {
        let mask = Ipv4AddrPrefix::mask(self.len as u32);
        u32::from_be_bytes(address.octets()) & mask == self.address_raw
    }
}

#[unstable(feature = "ip_prefix", issue = "86991")]
impl fmt::Display for Ipv4AddrPrefix {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{}/{}", Ipv4Addr::from_u32(self.address_raw), self.len)
    }
}

#[unstable(feature = "ip_prefix", issue = "86991")]
impl fmt::Debug for Ipv4AddrPrefix {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

#[unstable(feature = "ip_prefix", issue = "86991")]
impl From<Ipv4Addr> for Ipv4AddrPrefix {
    /// Converts an IPv4 address `a.b.c.d` to the prefix `a.b.c.d/32`.
    fn from(address: Ipv4Addr) -> Self {
        Ipv4AddrPrefix::new_unchecked(address, u32::BITS)
    }
}

impl Ipv6AddrPrefix {
    /// Creates a new IPv6 address prefix from an address and a prefix length.
    ///
    /// # Examples
    /// ```
    /// #![feature(ip_prefix)]
    ///
    /// use std::net::{Ipv6Addr, Ipv6AddrPrefix};
    ///
    /// // Create the address prefix `2001:db8::/32`.
    /// let prefix = Ipv6AddrPrefix::new(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0), 32).unwrap();
    ///
    /// // Error: Prefix length can not be more than 128 bits.
    /// Ipv6AddrPrefix::new(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1), 130).unwrap_err();
    /// ```
    #[unstable(feature = "ip_prefix", issue = "86991")]
    #[inline]
    pub const fn new(address: Ipv6Addr, len: u32) -> Result<Ipv6AddrPrefix, InvalidPrefixError> {
        if len <= u128::BITS {
            Ok(Ipv6AddrPrefix::new_unchecked(address, len))
        } else {
            Err(InvalidPrefixError { max_len: u128::BITS as u8 })
        }
    }

    // Private constructor that assumes len <= 128.
    // Useful because `Result::unwrap` is not yet usable in const contexts, so `new` can't be used.
    pub(crate) const fn new_unchecked(address: Ipv6Addr, len: u32) -> Ipv6AddrPrefix {
        let masked = {
            let mask = Ipv6AddrPrefix::mask(len);
            u128::from_be_bytes(address.octets()) & mask
        };

        Ipv6AddrPrefix { address_raw: masked, len: len as u8 }
    }

    /// Returns the address specifying this address prefix.
    ///
    /// The prefix address and the prefix length together uniquely define an address prefix.
    ///
    /// # Examples
    /// ```
    /// #![feature(ip_prefix)]
    ///
    /// use std::net::{Ipv6Addr, Ipv6AddrPrefix};
    ///
    /// // `2001:db8::/32`
    /// let prefix = Ipv6AddrPrefix::new(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 7), 32).unwrap();
    ///
    /// // Note that the address can be different than the address the
    /// // prefix was constructed with. The returned address only contains the bits
    /// // specified by the prefix length.
    /// assert_eq!(prefix.address(), Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0));
    /// assert_ne!(prefix.address(), Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 7));
    /// ```
    #[unstable(feature = "ip_prefix", issue = "86991")]
    #[inline]
    pub const fn address(&self) -> Ipv6Addr {
        Ipv6Addr::from_u128(self.address_raw)
    }

    /// Returns the prefix length of this address prefix.
    ///
    /// The prefix address and the prefix length together uniquely define an address prefix.
    ///
    /// # Examples
    /// ```
    /// #![feature(ip_prefix)]
    ///
    /// use std::net::{Ipv6Addr, Ipv6AddrPrefix};
    ///
    /// // `2001:db8::/32`
    /// let prefix = Ipv6AddrPrefix::new(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0), 32).unwrap();
    /// assert_eq!(prefix.len(), 32);
    /// ```
    #[unstable(feature = "ip_prefix", issue = "86991")]
    #[inline]
    pub const fn len(&self) -> u32 {
        self.len as u32
    }

    // Compute the bitmask specified by a prefix length.
    const fn mask(len: u32) -> u128 {
        if len == 0 {
            0
        } else {
            // shift will not overflow as len > 0, so u128::BITS - len < 128
            u128::MAX << (u128::BITS - len)
        }
    }

    /// Returns `true` if the given address is contained in the network described by this prefix,
    /// meaning that it starts with the same bits as the prefix.
    ///
    /// # Examples
    /// ```
    /// #![feature(ip_prefix)]
    ///
    /// use std::net::{Ipv6Addr, Ipv6AddrPrefix};
    ///
    /// // `2001:db8::/32`
    /// let prefix = Ipv6AddrPrefix::new(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0), 32).unwrap();
    ///
    /// assert_eq!(prefix.contains(&Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 7)), true);
    /// assert_eq!(prefix.contains(&Ipv6Addr::new(0x2001, 0xdb9, 0, 0, 0, 0, 0, 0)), false);
    /// ```
    #[unstable(feature = "ip_prefix", issue = "86991")]
    #[inline]
    pub const fn contains(&self, address: &Ipv6Addr) -> bool {
        let mask = Ipv6AddrPrefix::mask(self.len as u32);
        u128::from_be_bytes(address.octets()) & mask == self.address_raw
    }
}

#[unstable(feature = "ip_prefix", issue = "86991")]
impl fmt::Display for Ipv6AddrPrefix {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{}/{}", Ipv6Addr::from_u128(self.address_raw), self.len)
    }
}

#[unstable(feature = "ip_prefix", issue = "86991")]
impl fmt::Debug for Ipv6AddrPrefix {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

#[unstable(feature = "ip_prefix", issue = "86991")]
impl From<Ipv6Addr> for Ipv6AddrPrefix {
    /// Converts an IPv6 address `a:b:c:d:e:f:g:h` to the prefix `a:b:c:d:e:f:g:h/128`.
    fn from(address: Ipv6Addr) -> Self {
        Ipv6AddrPrefix::new_unchecked(address, u128::BITS)
    }
}

#[unstable(feature = "ip_prefix", issue = "86991")]
pub struct InvalidPrefixError {
    max_len: u8,
}

#[unstable(feature = "ip_prefix", issue = "86991")]
impl Error for InvalidPrefixError {}

#[unstable(feature = "ip_prefix", issue = "86991")]
impl fmt::Display for InvalidPrefixError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "Prefix length can not be more than {} bits.", self.max_len)
    }
}

#[unstable(feature = "ip_prefix", issue = "86991")]
impl fmt::Debug for InvalidPrefixError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("InvalidPrefixError").finish_non_exhaustive()
    }
}
