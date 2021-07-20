//! A private parser implementation of IPv4, IPv6, and socket addresses.
//!
//! This module is "publicly exported" through the `FromStr` implementations
//! below.

#[cfg(test)]
mod tests;

use crate::convert::TryInto as _;
use crate::error::Error;
use crate::fmt;
use crate::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use crate::str::FromStr;

trait ReadNumberHelper: crate::marker::Sized {
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
    fn new(input: &'a str) -> Parser<'a> {
        Parser { state: input.as_bytes() }
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
    fn parse_with<T, F>(&mut self, inner: F) -> Result<T, AddrParseError>
    where
        F: FnOnce(&mut Parser<'_>) -> Option<T>,
    {
        let result = inner(self);
        if self.state.is_empty() { result } else { None }.ok_or(AddrParseError(()))
    }

    /// Peek the next character from the input
    fn peek_char(&self) -> Option<char> {
        self.state.first().map(|&b| char::from(b))
    }

    /// Read the next character from the input
    fn read_char(&mut self) -> Option<char> {
        self.state.split_first().map(|(&b, tail)| {
            self.state = tail;
            char::from(b)
        })
    }

    #[must_use]
    /// Read the next character from the input if it matches the target.
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
    fn read_number<T: ReadNumberHelper>(
        &mut self,
        radix: u32,
        max_digits: Option<usize>,
        allow_zero_prefix: bool,
    ) -> Option<T> {
        self.read_atomically(move |p| {
            let mut result = T::ZERO;
            let mut digit_count = 0;
            let has_leading_zero = p.peek_char() == Some('0');

            while let Some(digit) = p.read_atomically(|p| p.read_char()?.to_digit(radix)) {
                result = result.checked_mul(radix)?;
                result = result.checked_add(digit)?;
                digit_count += 1;
                if let Some(max_digits) = max_digits {
                    if digit_count > max_digits {
                        return None;
                    }
                }
            }

            if digit_count == 0 {
                None
            } else if !allow_zero_prefix && has_leading_zero && digit_count > 1 {
                None
            } else {
                Some(result)
            }
        })
    }

    /// Read an IPv4 address.
    fn read_ipv4_addr(&mut self) -> Option<Ipv4Addr> {
        self.read_atomically(|p| {
            let mut groups = [0; 4];

            for (i, slot) in groups.iter_mut().enumerate() {
                *slot = p.read_separator('.', i, |p| {
                    // Disallow octal number in IP string.
                    // https://tools.ietf.org/html/rfc6943#section-3.1.1
                    p.read_number(10, None, false)
                })?;
            }

            Some(groups.into())
        })
    }

    /// Read an IPv6 Address.
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

    /// Read an IP Address, either IPv4 or IPv6.
    fn read_ip_addr(&mut self) -> Option<IpAddr> {
        self.read_ipv4_addr().map(IpAddr::V4).or_else(move || self.read_ipv6_addr().map(IpAddr::V6))
    }

    /// Read a `:` followed by a port in base 10.
    fn read_port(&mut self) -> Option<u16> {
        self.read_atomically(|p| {
            p.read_given_char(':')?;
            p.read_number(10, None, true)
        })
    }

    /// Read a `%` followed by a scope ID in base 10.
    fn read_scope_id(&mut self) -> Option<u32> {
        self.read_atomically(|p| {
            p.read_given_char('%')?;
            p.read_number(10, None, true)
        })
    }

    /// Read an IPv4 address with a port.
    fn read_socket_addr_v4(&mut self) -> Option<SocketAddrV4> {
        self.read_atomically(|p| {
            let ip = p.read_ipv4_addr()?;
            let port = p.read_port()?;
            Some(SocketAddrV4::new(ip, port))
        })
    }

    /// Read an IPv6 address with a port.
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

    /// Read an IP address with a port
    fn read_socket_addr(&mut self) -> Option<SocketAddr> {
        self.read_socket_addr_v4()
            .map(SocketAddr::V4)
            .or_else(|| self.read_socket_addr_v6().map(SocketAddr::V6))
    }
}

#[stable(feature = "ip_addr", since = "1.7.0")]
impl FromStr for IpAddr {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<IpAddr, AddrParseError> {
        Parser::new(s).parse_with(|p| p.read_ip_addr())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromStr for Ipv4Addr {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<Ipv4Addr, AddrParseError> {
        // don't try to parse if too long
        if s.len() > 15 {
            Err(AddrParseError(()))
        } else {
            Parser::new(s).parse_with(|p| p.read_ipv4_addr())
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromStr for Ipv6Addr {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<Ipv6Addr, AddrParseError> {
        Parser::new(s).parse_with(|p| p.read_ipv6_addr())
    }
}

#[stable(feature = "socket_addr_from_str", since = "1.5.0")]
impl FromStr for SocketAddrV4 {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<SocketAddrV4, AddrParseError> {
        Parser::new(s).parse_with(|p| p.read_socket_addr_v4())
    }
}

#[stable(feature = "socket_addr_from_str", since = "1.5.0")]
impl FromStr for SocketAddrV6 {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<SocketAddrV6, AddrParseError> {
        Parser::new(s).parse_with(|p| p.read_socket_addr_v6())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromStr for SocketAddr {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<SocketAddr, AddrParseError> {
        Parser::new(s).parse_with(|p| p.read_socket_addr())
    }
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
pub struct AddrParseError(());

#[stable(feature = "addr_parse_error_error", since = "1.4.0")]
impl fmt::Display for AddrParseError {
    #[allow(deprecated, deprecated_in_future)]
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(self.description())
    }
}

#[stable(feature = "addr_parse_error_error", since = "1.4.0")]
impl Error for AddrParseError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "invalid IP address syntax"
    }
}
