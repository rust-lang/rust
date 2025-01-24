use super::display_buffer::DisplayBuffer;
use crate::fmt::{self, Write};
use crate::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// An internet socket address, either IPv4 or IPv6.
///
/// Internet socket addresses consist of an [IP address], a 16-bit port number, as well
/// as possibly some version-dependent additional information. See [`SocketAddrV4`]'s and
/// [`SocketAddrV6`]'s respective documentation for more details.
///
/// The size of a `SocketAddr` instance may vary depending on the target operating
/// system.
///
/// [IP address]: IpAddr
///
/// # Examples
///
/// ```
/// use std::net::{IpAddr, Ipv4Addr, SocketAddr};
///
/// let socket = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
///
/// assert_eq!("127.0.0.1:8080".parse(), Ok(socket));
/// assert_eq!(socket.port(), 8080);
/// assert_eq!(socket.is_ipv4(), true);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum SocketAddr {
    /// An IPv4 socket address.
    #[stable(feature = "rust1", since = "1.0.0")]
    V4(#[stable(feature = "rust1", since = "1.0.0")] SocketAddrV4),
    /// An IPv6 socket address.
    #[stable(feature = "rust1", since = "1.0.0")]
    V6(#[stable(feature = "rust1", since = "1.0.0")] SocketAddrV6),
}

/// An IPv4 socket address.
///
/// IPv4 socket addresses consist of an [`IPv4` address] and a 16-bit port number, as
/// stated in [IETF RFC 793].
///
/// See [`SocketAddr`] for a type encompassing both IPv4 and IPv6 socket addresses.
///
/// The size of a `SocketAddrV4` struct may vary depending on the target operating
/// system. Do not assume that this type has the same memory layout as the underlying
/// system representation.
///
/// [IETF RFC 793]: https://tools.ietf.org/html/rfc793
/// [`IPv4` address]: Ipv4Addr
///
/// # Textual representation
///
/// `SocketAddrV4` provides a [`FromStr`](crate::str::FromStr) implementation.
/// It accepts an IPv4 address in its [textual representation], followed by a
/// single `:`, followed by the port encoded as a decimal integer.  Other
/// formats are not accepted.
///
/// [textual representation]: Ipv4Addr#textual-representation
///
/// # Examples
///
/// ```
/// use std::net::{Ipv4Addr, SocketAddrV4};
///
/// let socket = SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 8080);
///
/// assert_eq!("127.0.0.1:8080".parse(), Ok(socket));
/// assert_eq!(socket.ip(), &Ipv4Addr::new(127, 0, 0, 1));
/// assert_eq!(socket.port(), 8080);
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SocketAddrV4 {
    ip: Ipv4Addr,
    port: u16,
}

/// An IPv6 socket address.
///
/// IPv6 socket addresses consist of an [`IPv6` address], a 16-bit port number, as well
/// as fields containing the traffic class, the flow label, and a scope identifier
/// (see [IETF RFC 2553, Section 3.3] for more details).
///
/// See [`SocketAddr`] for a type encompassing both IPv4 and IPv6 socket addresses.
///
/// The size of a `SocketAddrV6` struct may vary depending on the target operating
/// system. Do not assume that this type has the same memory layout as the underlying
/// system representation.
///
/// [IETF RFC 2553, Section 3.3]: https://tools.ietf.org/html/rfc2553#section-3.3
/// [`IPv6` address]: Ipv6Addr
///
/// # Textual representation
///
/// `SocketAddrV6` provides a [`FromStr`](crate::str::FromStr) implementation,
/// based on the bracketed format recommended by [IETF RFC 5952],
/// with scope identifiers based on those specified in [IETF RFC 4007].
///
/// It accepts addresses consisting of the following elements, in order:
///   - A left square bracket (`[`)
///   - The [textual representation] of an IPv6 address
///   - _Optionally_, a percent sign (`%`) followed by the scope identifier
///     encoded as a decimal integer
///   - A right square bracket (`]`)
///   - A colon (`:`)
///   - The port, encoded as a decimal integer.
///
/// For example, the string `[2001:db8::413]:443` represents a `SocketAddrV6`
/// with the address `2001:db8::413` and port `443`.  The string
/// `[2001:db8::413%612]:443` represents the same address and port, with a
/// scope identifier of `612`.
///
/// Other formats are not accepted.
///
/// [IETF RFC 5952]: https://tools.ietf.org/html/rfc5952#section-6
/// [IETF RFC 4007]: https://tools.ietf.org/html/rfc4007#section-11
/// [textual representation]: Ipv6Addr#textual-representation
///
/// # Examples
///
/// ```
/// use std::net::{Ipv6Addr, SocketAddrV6};
///
/// let socket = SocketAddrV6::new(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1), 8080, 0, 0);
///
/// assert_eq!("[2001:db8::1]:8080".parse(), Ok(socket));
/// assert_eq!(socket.ip(), &Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1));
/// assert_eq!(socket.port(), 8080);
///
/// let mut with_scope = socket.clone();
/// with_scope.set_scope_id(3);
/// assert_eq!("[2001:db8::1%3]:8080".parse(), Ok(with_scope));
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SocketAddrV6 {
    ip: Ipv6Addr,
    port: u16,
    flowinfo: u32,
    scope_id: u32,
}

impl SocketAddr {
    /// Creates a new socket address from an [IP address] and a port number.
    ///
    /// [IP address]: IpAddr
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    ///
    /// let socket = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
    /// assert_eq!(socket.ip(), IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
    /// assert_eq!(socket.port(), 8080);
    /// ```
    #[stable(feature = "ip_addr", since = "1.7.0")]
    #[must_use]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn new(ip: IpAddr, port: u16) -> SocketAddr {
        match ip {
            IpAddr::V4(a) => SocketAddr::V4(SocketAddrV4::new(a, port)),
            IpAddr::V6(a) => SocketAddr::V6(SocketAddrV6::new(a, port, 0, 0)),
        }
    }

    /// Returns the IP address associated with this socket address.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    ///
    /// let socket = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
    /// assert_eq!(socket.ip(), IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
    /// ```
    #[must_use]
    #[stable(feature = "ip_addr", since = "1.7.0")]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn ip(&self) -> IpAddr {
        match *self {
            SocketAddr::V4(ref a) => IpAddr::V4(*a.ip()),
            SocketAddr::V6(ref a) => IpAddr::V6(*a.ip()),
        }
    }

    /// Changes the IP address associated with this socket address.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    ///
    /// let mut socket = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
    /// socket.set_ip(IpAddr::V4(Ipv4Addr::new(10, 10, 0, 1)));
    /// assert_eq!(socket.ip(), IpAddr::V4(Ipv4Addr::new(10, 10, 0, 1)));
    /// ```
    #[inline]
    #[stable(feature = "sockaddr_setters", since = "1.9.0")]
    #[rustc_const_unstable(feature = "const_sockaddr_setters", issue = "131714")]
    pub const fn set_ip(&mut self, new_ip: IpAddr) {
        // `match (*self, new_ip)` would have us mutate a copy of self only to throw it away.
        match (self, new_ip) {
            (&mut SocketAddr::V4(ref mut a), IpAddr::V4(new_ip)) => a.set_ip(new_ip),
            (&mut SocketAddr::V6(ref mut a), IpAddr::V6(new_ip)) => a.set_ip(new_ip),
            (self_, new_ip) => *self_ = Self::new(new_ip, self_.port()),
        }
    }

    /// Returns the port number associated with this socket address.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    ///
    /// let socket = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
    /// assert_eq!(socket.port(), 8080);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn port(&self) -> u16 {
        match *self {
            SocketAddr::V4(ref a) => a.port(),
            SocketAddr::V6(ref a) => a.port(),
        }
    }

    /// Changes the port number associated with this socket address.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    ///
    /// let mut socket = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
    /// socket.set_port(1025);
    /// assert_eq!(socket.port(), 1025);
    /// ```
    #[inline]
    #[stable(feature = "sockaddr_setters", since = "1.9.0")]
    #[rustc_const_unstable(feature = "const_sockaddr_setters", issue = "131714")]
    pub const fn set_port(&mut self, new_port: u16) {
        match *self {
            SocketAddr::V4(ref mut a) => a.set_port(new_port),
            SocketAddr::V6(ref mut a) => a.set_port(new_port),
        }
    }

    /// Returns [`true`] if the [IP address] in this `SocketAddr` is an
    /// [`IPv4` address], and [`false`] otherwise.
    ///
    /// [IP address]: IpAddr
    /// [`IPv4` address]: IpAddr::V4
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    ///
    /// let socket = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
    /// assert_eq!(socket.is_ipv4(), true);
    /// assert_eq!(socket.is_ipv6(), false);
    /// ```
    #[must_use]
    #[stable(feature = "sockaddr_checker", since = "1.16.0")]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn is_ipv4(&self) -> bool {
        matches!(*self, SocketAddr::V4(_))
    }

    /// Returns [`true`] if the [IP address] in this `SocketAddr` is an
    /// [`IPv6` address], and [`false`] otherwise.
    ///
    /// [IP address]: IpAddr
    /// [`IPv6` address]: IpAddr::V6
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{IpAddr, Ipv6Addr, SocketAddr};
    ///
    /// let socket = SocketAddr::new(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 65535, 0, 1)), 8080);
    /// assert_eq!(socket.is_ipv4(), false);
    /// assert_eq!(socket.is_ipv6(), true);
    /// ```
    #[must_use]
    #[stable(feature = "sockaddr_checker", since = "1.16.0")]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn is_ipv6(&self) -> bool {
        matches!(*self, SocketAddr::V6(_))
    }
}

impl SocketAddrV4 {
    /// Creates a new socket address from an [`IPv4` address] and a port number.
    ///
    /// [`IPv4` address]: Ipv4Addr
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV4, Ipv4Addr};
    ///
    /// let socket = SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 8080);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn new(ip: Ipv4Addr, port: u16) -> SocketAddrV4 {
        SocketAddrV4 { ip, port }
    }

    /// Returns the IP address associated with this socket address.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV4, Ipv4Addr};
    ///
    /// let socket = SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 8080);
    /// assert_eq!(socket.ip(), &Ipv4Addr::new(127, 0, 0, 1));
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn ip(&self) -> &Ipv4Addr {
        &self.ip
    }

    /// Changes the IP address associated with this socket address.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV4, Ipv4Addr};
    ///
    /// let mut socket = SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 8080);
    /// socket.set_ip(Ipv4Addr::new(192, 168, 0, 1));
    /// assert_eq!(socket.ip(), &Ipv4Addr::new(192, 168, 0, 1));
    /// ```
    #[inline]
    #[stable(feature = "sockaddr_setters", since = "1.9.0")]
    #[rustc_const_unstable(feature = "const_sockaddr_setters", issue = "131714")]
    pub const fn set_ip(&mut self, new_ip: Ipv4Addr) {
        self.ip = new_ip;
    }

    /// Returns the port number associated with this socket address.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV4, Ipv4Addr};
    ///
    /// let socket = SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 8080);
    /// assert_eq!(socket.port(), 8080);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn port(&self) -> u16 {
        self.port
    }

    /// Changes the port number associated with this socket address.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV4, Ipv4Addr};
    ///
    /// let mut socket = SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 8080);
    /// socket.set_port(4242);
    /// assert_eq!(socket.port(), 4242);
    /// ```
    #[inline]
    #[stable(feature = "sockaddr_setters", since = "1.9.0")]
    #[rustc_const_unstable(feature = "const_sockaddr_setters", issue = "131714")]
    pub const fn set_port(&mut self, new_port: u16) {
        self.port = new_port;
    }
}

impl SocketAddrV6 {
    /// Creates a new socket address from an [`IPv6` address], a 16-bit port number,
    /// and the `flowinfo` and `scope_id` fields.
    ///
    /// For more information on the meaning and layout of the `flowinfo` and `scope_id`
    /// parameters, see [IETF RFC 2553, Section 3.3].
    ///
    /// [IETF RFC 2553, Section 3.3]: https://tools.ietf.org/html/rfc2553#section-3.3
    /// [`IPv6` address]: Ipv6Addr
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV6, Ipv6Addr};
    ///
    /// let socket = SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1), 8080, 0, 0);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn new(ip: Ipv6Addr, port: u16, flowinfo: u32, scope_id: u32) -> SocketAddrV6 {
        SocketAddrV6 { ip, port, flowinfo, scope_id }
    }

    /// Returns the IP address associated with this socket address.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV6, Ipv6Addr};
    ///
    /// let socket = SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1), 8080, 0, 0);
    /// assert_eq!(socket.ip(), &Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1));
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn ip(&self) -> &Ipv6Addr {
        &self.ip
    }

    /// Changes the IP address associated with this socket address.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV6, Ipv6Addr};
    ///
    /// let mut socket = SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1), 8080, 0, 0);
    /// socket.set_ip(Ipv6Addr::new(76, 45, 0, 0, 0, 0, 0, 0));
    /// assert_eq!(socket.ip(), &Ipv6Addr::new(76, 45, 0, 0, 0, 0, 0, 0));
    /// ```
    #[inline]
    #[stable(feature = "sockaddr_setters", since = "1.9.0")]
    #[rustc_const_unstable(feature = "const_sockaddr_setters", issue = "131714")]
    pub const fn set_ip(&mut self, new_ip: Ipv6Addr) {
        self.ip = new_ip;
    }

    /// Returns the port number associated with this socket address.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV6, Ipv6Addr};
    ///
    /// let socket = SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1), 8080, 0, 0);
    /// assert_eq!(socket.port(), 8080);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn port(&self) -> u16 {
        self.port
    }

    /// Changes the port number associated with this socket address.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV6, Ipv6Addr};
    ///
    /// let mut socket = SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1), 8080, 0, 0);
    /// socket.set_port(4242);
    /// assert_eq!(socket.port(), 4242);
    /// ```
    #[inline]
    #[stable(feature = "sockaddr_setters", since = "1.9.0")]
    #[rustc_const_unstable(feature = "const_sockaddr_setters", issue = "131714")]
    pub const fn set_port(&mut self, new_port: u16) {
        self.port = new_port;
    }

    /// Returns the flow information associated with this address.
    ///
    /// This information corresponds to the `sin6_flowinfo` field in C's `netinet/in.h`,
    /// as specified in [IETF RFC 2553, Section 3.3].
    /// It combines information about the flow label and the traffic class as specified
    /// in [IETF RFC 2460], respectively [Section 6] and [Section 7].
    ///
    /// [IETF RFC 2553, Section 3.3]: https://tools.ietf.org/html/rfc2553#section-3.3
    /// [IETF RFC 2460]: https://tools.ietf.org/html/rfc2460
    /// [Section 6]: https://tools.ietf.org/html/rfc2460#section-6
    /// [Section 7]: https://tools.ietf.org/html/rfc2460#section-7
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV6, Ipv6Addr};
    ///
    /// let socket = SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1), 8080, 10, 0);
    /// assert_eq!(socket.flowinfo(), 10);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn flowinfo(&self) -> u32 {
        self.flowinfo
    }

    /// Changes the flow information associated with this socket address.
    ///
    /// See [`SocketAddrV6::flowinfo`]'s documentation for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV6, Ipv6Addr};
    ///
    /// let mut socket = SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1), 8080, 10, 0);
    /// socket.set_flowinfo(56);
    /// assert_eq!(socket.flowinfo(), 56);
    /// ```
    #[inline]
    #[stable(feature = "sockaddr_setters", since = "1.9.0")]
    #[rustc_const_unstable(feature = "const_sockaddr_setters", issue = "131714")]
    pub const fn set_flowinfo(&mut self, new_flowinfo: u32) {
        self.flowinfo = new_flowinfo;
    }

    /// Returns the scope ID associated with this address.
    ///
    /// This information corresponds to the `sin6_scope_id` field in C's `netinet/in.h`,
    /// as specified in [IETF RFC 2553, Section 3.3].
    ///
    /// [IETF RFC 2553, Section 3.3]: https://tools.ietf.org/html/rfc2553#section-3.3
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV6, Ipv6Addr};
    ///
    /// let socket = SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1), 8080, 0, 78);
    /// assert_eq!(socket.scope_id(), 78);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_socketaddr", since = "1.69.0")]
    #[inline]
    pub const fn scope_id(&self) -> u32 {
        self.scope_id
    }

    /// Changes the scope ID associated with this socket address.
    ///
    /// See [`SocketAddrV6::scope_id`]'s documentation for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::net::{SocketAddrV6, Ipv6Addr};
    ///
    /// let mut socket = SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1), 8080, 0, 78);
    /// socket.set_scope_id(42);
    /// assert_eq!(socket.scope_id(), 42);
    /// ```
    #[inline]
    #[stable(feature = "sockaddr_setters", since = "1.9.0")]
    #[rustc_const_unstable(feature = "const_sockaddr_setters", issue = "131714")]
    pub const fn set_scope_id(&mut self, new_scope_id: u32) {
        self.scope_id = new_scope_id;
    }
}

#[stable(feature = "ip_from_ip", since = "1.16.0")]
impl From<SocketAddrV4> for SocketAddr {
    /// Converts a [`SocketAddrV4`] into a [`SocketAddr::V4`].
    #[inline]
    fn from(sock4: SocketAddrV4) -> SocketAddr {
        SocketAddr::V4(sock4)
    }
}

#[stable(feature = "ip_from_ip", since = "1.16.0")]
impl From<SocketAddrV6> for SocketAddr {
    /// Converts a [`SocketAddrV6`] into a [`SocketAddr::V6`].
    #[inline]
    fn from(sock6: SocketAddrV6) -> SocketAddr {
        SocketAddr::V6(sock6)
    }
}

#[stable(feature = "addr_from_into_ip", since = "1.17.0")]
impl<I: Into<IpAddr>> From<(I, u16)> for SocketAddr {
    /// Converts a tuple struct (Into<[`IpAddr`]>, `u16`) into a [`SocketAddr`].
    ///
    /// This conversion creates a [`SocketAddr::V4`] for an [`IpAddr::V4`]
    /// and creates a [`SocketAddr::V6`] for an [`IpAddr::V6`].
    ///
    /// `u16` is treated as port of the newly created [`SocketAddr`].
    fn from(pieces: (I, u16)) -> SocketAddr {
        SocketAddr::new(pieces.0.into(), pieces.1)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for SocketAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            SocketAddr::V4(ref a) => a.fmt(f),
            SocketAddr::V6(ref a) => a.fmt(f),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for SocketAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for SocketAddrV4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // If there are no alignment requirements, write the socket address directly to `f`.
        // Otherwise, write it to a local buffer and then use `f.pad`.
        if f.precision().is_none() && f.width().is_none() {
            write!(f, "{}:{}", self.ip(), self.port())
        } else {
            const LONGEST_IPV4_SOCKET_ADDR: &str = "255.255.255.255:65535";

            let mut buf = DisplayBuffer::<{ LONGEST_IPV4_SOCKET_ADDR.len() }>::new();
            // Buffer is long enough for the longest possible IPv4 socket address, so this should never fail.
            write!(buf, "{}:{}", self.ip(), self.port()).unwrap();

            f.pad(buf.as_str())
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for SocketAddrV4 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for SocketAddrV6 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // If there are no alignment requirements, write the socket address directly to `f`.
        // Otherwise, write it to a local buffer and then use `f.pad`.
        if f.precision().is_none() && f.width().is_none() {
            match self.scope_id() {
                0 => write!(f, "[{}]:{}", self.ip(), self.port()),
                scope_id => write!(f, "[{}%{}]:{}", self.ip(), scope_id, self.port()),
            }
        } else {
            const LONGEST_IPV6_SOCKET_ADDR: &str =
                "[ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff%4294967295]:65535";

            let mut buf = DisplayBuffer::<{ LONGEST_IPV6_SOCKET_ADDR.len() }>::new();
            match self.scope_id() {
                0 => write!(buf, "[{}]:{}", self.ip(), self.port()),
                scope_id => write!(buf, "[{}%{}]:{}", self.ip(), scope_id, self.port()),
            }
            // Buffer is long enough for the longest possible IPv6 socket address, so this should never fail.
            .unwrap();

            f.pad(buf.as_str())
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for SocketAddrV6 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}
