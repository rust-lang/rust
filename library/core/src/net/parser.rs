//! A private parser implementation of IPv4, IPv6, and socket addresses.
//!
//! This module is "publicly exported" through the `FromStr` implementations
//! below.

use crate::error::Error;
use crate::fmt;
use crate::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use crate::str::FromStr;

trait ReadNumberHelper: Sized {
    const ZERO: Self;
    fn checked_mul(&self, other: u32) -> Option<Self>;
    fn checked_add(&self, other: u32) -> Option<Self>;
}

macro_rules! impl_helper {
    ($($t:ty)*) => ($(impl ReadNumberHelper for $t {
        const ZERO: Self = 0;
        #[inline]
        fn checked_mul(&self, other: u32) -> Option<Self> {
            Self::checked_mul(*self, other.try_into().ok()?)
        }
        #[inline]
        fn checked_add(&self, other: u32) -> Option<Self> {
            Self::checked_add(*self, other.try_into().ok()?)
        }
    })*)
}

impl_helper! { u8 u16 u32 }

struct Parser<'a> {
    // Parsing as ASCII, so can use byte array.
    state: &'a [u8],
}

impl<'a> Parser<'a> {
    fn new(input: &'a [u8]) -> Parser<'a> {
        Parser { state: input }
    }

    /// Run a parser, and restore the pre-parse state if it fails.
    fn read_atomically<T, F>(&mut self, inner: F) -> Option<T>
    where
        F: FnOnce(&mut Parser<'_>) -> Option<T>,
    {
        let state = self.state;
        let result = inner(self);
        if result.is_none() {
            self.state = state;
        }
        result
    }

    /// Run a parser, but fail if the entire input wasn't consumed.
    /// Doesn't run atomically.
    fn parse_with<T, F>(&mut self, inner: F, kind: AddrKind) -> Result<T, AddrParseError>
    where
        F: FnOnce(&mut Parser<'_>) -> Option<T>,
    {
        let result = inner(self);
        if self.state.is_empty() { result } else { None }.ok_or(AddrParseError(kind))
    }

    /// Peek the next character from the input
    fn peek_char(&self) -> Option<char> {
        self.state.first().map(|&b| char::from(b))
    }

    /// Reads the next character from the input
    fn read_char(&mut self) -> Option<char> {
        self.state.split_first().map(|(&b, tail)| {
            self.state = tail;
            char::from(b)
        })
    }

    #[must_use]
    /// Reads the next character from the input if it matches the target.
    fn read_given_char(&mut self, target: char) -> Option<()> {
        self.read_atomically(|p| {
            p.read_char().and_then(|c| if c == target { Some(()) } else { None })
        })
    }

    /// Helper for reading separators in an indexed loop. Reads the separator
    /// character iff index > 0, then runs the parser. When used in a loop,
    /// the separator character will only be read on index > 0 (see
    /// read_ipv4_addr for an example)
    fn read_separator<T, F>(&mut self, sep: char, index: usize, inner: F) -> Option<T>
    where
        F: FnOnce(&mut Parser<'_>) -> Option<T>,
    {
        self.read_atomically(move |p| {
            if index > 0 {
                p.read_given_char(sep)?;
            }
            inner(p)
        })
    }

    // Read a number off the front of the input in the given radix, stopping
    // at the first non-digit character or eof. Fails if the number has more
    // digits than max_digits or if there is no number.
    //
    // INVARIANT: `max_digits` must be less than the number of digits that `u32`
    // can represent.
    fn read_number<T: ReadNumberHelper + TryFrom<u32>>(
        &mut self,
        radix: u32,
        max_digits: Option<usize>,
        allow_zero_prefix: bool,
    ) -> Option<T> {
        self.read_atomically(move |p| {
            let mut digit_count = 0;
            let has_leading_zero = p.peek_char() == Some('0');

            // If max_digits.is_some(), then we are parsing a `u8` or `u16` and
            // don't need to use checked arithmetic since it fits within a `u32`.
            let result = if let Some(max_digits) = max_digits {
                // u32::MAX = 4_294_967_295u32, which is 10 digits long.
                // `max_digits` must be less than 10 to not overflow a `u32`.
                debug_assert!(max_digits < 10);

                let mut result = 0_u32;
                while let Some(digit) = p.read_atomically(|p| p.read_char()?.to_digit(radix)) {
                    result *= radix;
                    result += digit;
                    digit_count += 1;

                    if digit_count > max_digits {
                        return None;
                    }
                }

                result.try_into().ok()
            } else {
                let mut result = T::ZERO;

                while let Some(digit) = p.read_atomically(|p| p.read_char()?.to_digit(radix)) {
                    result = result.checked_mul(radix)?;
                    result = result.checked_add(digit)?;
                    digit_count += 1;
                }

                Some(result)
            };

            if digit_count == 0 {
                None
            } else if !allow_zero_prefix && has_leading_zero && digit_count > 1 {
                None
            } else {
                result
            }
        })
    }

    /// Reads an IPv4 address.
    fn read_ipv4_addr(&mut self) -> Option<Ipv4Addr> {
        self.read_atomically(|p| {
            let mut groups = [0; 4];

            for (i, slot) in groups.iter_mut().enumerate() {
                *slot = p.read_separator('.', i, |p| {
                    // Disallow octal number in IP string.
                    // https://tools.ietf.org/html/rfc6943#section-3.1.1
                    p.read_number(10, Some(3), false)
                })?;
            }

            Some(groups.into())
        })
    }

    /// Reads an IPv6 address.
    fn read_ipv6_addr(&mut self) -> Option<Ipv6Addr> {
        /// Read a chunk of an IPv6 address into `groups`. Returns the number
        /// of groups read, along with a bool indicating if an embedded
        /// trailing IPv4 address was read. Specifically, read a series of
        /// colon-separated IPv6 groups (0x0000 - 0xFFFF), with an optional
        /// trailing embedded IPv4 address.
        fn read_groups(p: &mut Parser<'_>, groups: &mut [u16]) -> (usize, bool) {
            let limit = groups.len();

            for (i, slot) in groups.iter_mut().enumerate() {
                // Try to read a trailing embedded IPv4 address. There must be
                // at least two groups left.
                if i < limit - 1 {
                    let ipv4 = p.read_separator(':', i, |p| p.read_ipv4_addr());

                    if let Some(v4_addr) = ipv4 {
                        let [one, two, three, four] = v4_addr.octets();
                        groups[i + 0] = u16::from_be_bytes([one, two]);
                        groups[i + 1] = u16::from_be_bytes([three, four]);
                        return (i + 2, true);
                    }
                }

                let group = p.read_separator(':', i, |p| p.read_number(16, Some(4), true));

                match group {
                    Some(g) => *slot = g,
                    None => return (i, false),
                }
            }
            (groups.len(), false)
        }

        self.read_atomically(|p| {
            // Read the front part of the address; either the whole thing, or up
            // to the first ::
            let mut head = [0; 8];
            let (head_size, head_ipv4) = read_groups(p, &mut head);

            if head_size == 8 {
                return Some(head.into());
            }

            // IPv4 part is not allowed before `::`
            if head_ipv4 {
                return None;
            }

            // Read `::` if previous code parsed less than 8 groups.
            // `::` indicates one or more groups of 16 bits of zeros.
            p.read_given_char(':')?;
            p.read_given_char(':')?;

            // Read the back part of the address. The :: must contain at least one
            // set of zeroes, so our max length is 7.
            let mut tail = [0; 7];
            let limit = 8 - (head_size + 1);
            let (tail_size, _) = read_groups(p, &mut tail[..limit]);

            // Concat the head and tail of the IP address
            head[(8 - tail_size)..8].copy_from_slice(&tail[..tail_size]);

            Some(head.into())
        })
    }

    /// Reads an IP address, either IPv4 or IPv6.
    fn read_ip_addr(&mut self) -> Option<IpAddr> {
        self.read_ipv4_addr().map(IpAddr::V4).or_else(move || self.read_ipv6_addr().map(IpAddr::V6))
    }

    /// Reads a `:` followed by a port in base 10.
    fn read_port(&mut self) -> Option<u16> {
        self.read_atomically(|p| {
            p.read_given_char(':')?;
            p.read_number(10, None, true)
        })
    }

    /// Reads a `%` followed by a scope ID in base 10.
    fn read_scope_id(&mut self) -> Option<u32> {
        self.read_atomically(|p| {
            p.read_given_char('%')?;
            p.read_number(10, None, true)
        })
    }

    /// Reads an IPv4 address with a port.
    fn read_socket_addr_v4(&mut self) -> Option<SocketAddrV4> {
        self.read_atomically(|p| {
            let ip = p.read_ipv4_addr()?;
            let port = p.read_port()?;
            Some(SocketAddrV4::new(ip, port))
        })
    }

    /// Reads an IPv6 address with a port.
    fn read_socket_addr_v6(&mut self) -> Option<SocketAddrV6> {
        self.read_atomically(|p| {
            p.read_given_char('[')?;
            let ip = p.read_ipv6_addr()?;
            let scope_id = p.read_scope_id().unwrap_or(0);
            p.read_given_char(']')?;

            let port = p.read_port()?;
            Some(SocketAddrV6::new(ip, port, 0, scope_id))
        })
    }

    /// Reads an IP address with a port.
    fn read_socket_addr(&mut self) -> Option<SocketAddr> {
        self.read_socket_addr_v4()
            .map(SocketAddr::V4)
            .or_else(|| self.read_socket_addr_v6().map(SocketAddr::V6))
    }
}

impl IpAddr {
    /// Parse an IP address from a slice of bytes.
    ///
    /// ```
    /// #![feature(addr_parse_ascii)]
    ///
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    ///
    /// let localhost_v4 = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
    /// let localhost_v6 = IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1));
    ///
    /// assert_eq!(IpAddr::parse_ascii(b"127.0.0.1"), Ok(localhost_v4));
    /// assert_eq!(IpAddr::parse_ascii(b"::1"), Ok(localhost_v6));
    /// ```
    #[unstable(feature = "addr_parse_ascii", issue = "101035")]
    pub fn parse_ascii(b: &[u8]) -> Result<Self, AddrParseError> {
        Parser::new(b).parse_with(|p| p.read_ip_addr(), AddrKind::Ip)
    }
}

#[stable(feature = "ip_addr", since = "1.7.0")]
impl FromStr for IpAddr {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<IpAddr, AddrParseError> {
        Self::parse_ascii(s.as_bytes())
    }
}

impl Ipv4Addr {
    /// Parse an IPv4 address from a slice of bytes.
    ///
    /// ```
    /// #![feature(addr_parse_ascii)]
    ///
    /// use std::net::Ipv4Addr;
    ///
    /// let localhost = Ipv4Addr::new(127, 0, 0, 1);
    ///
    /// assert_eq!(Ipv4Addr::parse_ascii(b"127.0.0.1"), Ok(localhost));
    /// ```
    #[unstable(feature = "addr_parse_ascii", issue = "101035")]
    pub fn parse_ascii(b: &[u8]) -> Result<Self, AddrParseError> {
        // don't try to parse if too long
        if b.len() > 15 {
            Err(AddrParseError(AddrKind::Ipv4))
        } else {
            Parser::new(b).parse_with(|p| p.read_ipv4_addr(), AddrKind::Ipv4)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromStr for Ipv4Addr {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<Ipv4Addr, AddrParseError> {
        Self::parse_ascii(s.as_bytes())
    }
}

impl Ipv6Addr {
    /// Parse an IPv6 address from a slice of bytes.
    ///
    /// ```
    /// #![feature(addr_parse_ascii)]
    ///
    /// use std::net::Ipv6Addr;
    ///
    /// let localhost = Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1);
    ///
    /// assert_eq!(Ipv6Addr::parse_ascii(b"::1"), Ok(localhost));
    /// ```
    #[unstable(feature = "addr_parse_ascii", issue = "101035")]
    pub fn parse_ascii(b: &[u8]) -> Result<Self, AddrParseError> {
        Parser::new(b).parse_with(|p| p.read_ipv6_addr(), AddrKind::Ipv6)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromStr for Ipv6Addr {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<Ipv6Addr, AddrParseError> {
        Self::parse_ascii(s.as_bytes())
    }
}

impl SocketAddrV4 {
    /// Parse an IPv4 socket address from a slice of bytes.
    ///
    /// ```
    /// #![feature(addr_parse_ascii)]
    ///
    /// use std::net::{Ipv4Addr, SocketAddrV4};
    ///
    /// let socket = SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 8080);
    ///
    /// assert_eq!(SocketAddrV4::parse_ascii(b"127.0.0.1:8080"), Ok(socket));
    /// ```
    #[unstable(feature = "addr_parse_ascii", issue = "101035")]
    pub fn parse_ascii(b: &[u8]) -> Result<Self, AddrParseError> {
        Parser::new(b).parse_with(|p| p.read_socket_addr_v4(), AddrKind::SocketV4)
    }
}

#[stable(feature = "socket_addr_from_str", since = "1.5.0")]
impl FromStr for SocketAddrV4 {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<SocketAddrV4, AddrParseError> {
        Self::parse_ascii(s.as_bytes())
    }
}

impl SocketAddrV6 {
    /// Parse an IPv6 socket address from a slice of bytes.
    ///
    /// ```
    /// #![feature(addr_parse_ascii)]
    ///
    /// use std::net::{Ipv6Addr, SocketAddrV6};
    ///
    /// let socket = SocketAddrV6::new(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1), 8080, 0, 0);
    ///
    /// assert_eq!(SocketAddrV6::parse_ascii(b"[2001:db8::1]:8080"), Ok(socket));
    /// ```
    #[unstable(feature = "addr_parse_ascii", issue = "101035")]
    pub fn parse_ascii(b: &[u8]) -> Result<Self, AddrParseError> {
        Parser::new(b).parse_with(|p| p.read_socket_addr_v6(), AddrKind::SocketV6)
    }
}

#[stable(feature = "socket_addr_from_str", since = "1.5.0")]
impl FromStr for SocketAddrV6 {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<SocketAddrV6, AddrParseError> {
        Self::parse_ascii(s.as_bytes())
    }
}

impl SocketAddr {
    /// Parse a socket address from a slice of bytes.
    ///
    /// ```
    /// #![feature(addr_parse_ascii)]
    ///
    /// use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
    ///
    /// let socket_v4 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
    /// let socket_v6 = SocketAddr::new(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1)), 8080);
    ///
    /// assert_eq!(SocketAddr::parse_ascii(b"127.0.0.1:8080"), Ok(socket_v4));
    /// assert_eq!(SocketAddr::parse_ascii(b"[::1]:8080"), Ok(socket_v6));
    /// ```
    #[unstable(feature = "addr_parse_ascii", issue = "101035")]
    pub fn parse_ascii(b: &[u8]) -> Result<Self, AddrParseError> {
        Parser::new(b).parse_with(|p| p.read_socket_addr(), AddrKind::Socket)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromStr for SocketAddr {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<SocketAddr, AddrParseError> {
        Self::parse_ascii(s.as_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AddrKind {
    Ip,
    Ipv4,
    Ipv6,
    Socket,
    SocketV4,
    SocketV6,
}

/// An error which can be returned when parsing an IP address or a socket address.
///
/// This error is used as the error type for the [`FromStr`] implementation for
/// [`IpAddr`], [`Ipv4Addr`], [`Ipv6Addr`], [`SocketAddr`], [`SocketAddrV4`], and
/// [`SocketAddrV6`].
///
/// # Potential causes
///
/// `AddrParseError` may be thrown because the provided string does not parse as the given type,
/// often because it includes information only handled by a different address type.
///
/// ```should_panic
/// use std::net::IpAddr;
/// let _foo: IpAddr = "127.0.0.1:8080".parse().expect("Cannot handle the socket port");
/// ```
///
/// [`IpAddr`] doesn't handle the port. Use [`SocketAddr`] instead.
///
/// ```
/// use std::net::SocketAddr;
///
/// // No problem, the `panic!` message has disappeared.
/// let _foo: SocketAddr = "127.0.0.1:8080".parse().expect("unreachable panic");
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddrParseError(AddrKind);

#[stable(feature = "addr_parse_error_error", since = "1.4.0")]
impl fmt::Display for AddrParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            AddrKind::Ip => "invalid IP address syntax",
            AddrKind::Ipv4 => "invalid IPv4 address syntax",
            AddrKind::Ipv6 => "invalid IPv6 address syntax",
            AddrKind::Socket => "invalid socket address syntax",
            AddrKind::SocketV4 => "invalid IPv4 socket address syntax",
            AddrKind::SocketV6 => "invalid IPv6 socket address syntax",
        }
        .fmt(f)
    }
}

#[stable(feature = "addr_parse_error_error", since = "1.4.0")]
impl Error for AddrParseError {}
