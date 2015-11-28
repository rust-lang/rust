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

use fmt;
use hash;
use io;
use mem;
use net::{lookup_host, ntoh, hton, IpAddr, Ipv4Addr, Ipv6Addr};
use option;
use sys::net::netc as c;
use sys_common::{FromInner, AsInner, IntoInner};
use vec;

/// Representation of a socket address for networking applications.
///
/// A socket address can either represent the IPv4 or IPv6 protocol and is
/// paired with at least a port number as well. Each protocol may have more
/// specific information about the address available to it as well.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum SocketAddr {
    /// An IPv4 socket address which is a (ip, port) combination.
    #[stable(feature = "rust1", since = "1.0.0")]
    V4(SocketAddrV4),
    /// An IPv6 socket address
    #[stable(feature = "rust1", since = "1.0.0")]
    V6(SocketAddrV6),
}

/// An IPv4 socket address which is a (ip, port) combination.
#[derive(Copy)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SocketAddrV4 { inner: c::sockaddr_in }

/// An IPv6 socket address.
#[derive(Copy)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SocketAddrV6 { inner: c::sockaddr_in6 }

impl SocketAddr {
    /// Creates a new socket address from the (ip, port) pair.
    #[unstable(feature = "ip_addr", reason = "recent addition", issue = "27801")]
    pub fn new(ip: IpAddr, port: u16) -> SocketAddr {
        match ip {
            IpAddr::V4(a) => SocketAddr::V4(SocketAddrV4::new(a, port)),
            IpAddr::V6(a) => SocketAddr::V6(SocketAddrV6::new(a, port, 0, 0)),
        }
    }

    /// Returns the IP address associated with this socket address.
    #[unstable(feature = "ip_addr", reason = "recent addition", issue = "27801")]
    pub fn ip(&self) -> IpAddr {
        match *self {
            SocketAddr::V4(ref a) => IpAddr::V4(*a.ip()),
            SocketAddr::V6(ref a) => IpAddr::V6(*a.ip()),
        }
    }

    /// Returns the port number associated with this socket address.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn port(&self) -> u16 {
        match *self {
            SocketAddr::V4(ref a) => a.port(),
            SocketAddr::V6(ref a) => a.port(),
        }
    }
}

impl SocketAddrV4 {
    /// Creates a new socket address from the (ip, port) pair.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(ip: Ipv4Addr, port: u16) -> SocketAddrV4 {
        SocketAddrV4 {
            inner: c::sockaddr_in {
                sin_family: c::AF_INET as c::sa_family_t,
                sin_port: hton(port),
                sin_addr: *ip.as_inner(),
                .. unsafe { mem::zeroed() }
            },
        }
    }

    /// Returns the IP address associated with this socket address.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn ip(&self) -> &Ipv4Addr {
        unsafe {
            &*(&self.inner.sin_addr as *const c::in_addr as *const Ipv4Addr)
        }
    }

    /// Returns the port number associated with this socket address.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn port(&self) -> u16 { ntoh(self.inner.sin_port) }
}

impl SocketAddrV6 {
    /// Creates a new socket address from the ip/port/flowinfo/scope_id
    /// components.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(ip: Ipv6Addr, port: u16, flowinfo: u32, scope_id: u32)
               -> SocketAddrV6 {
        SocketAddrV6 {
            inner: c::sockaddr_in6 {
                sin6_family: c::AF_INET6 as c::sa_family_t,
                sin6_port: hton(port),
                sin6_addr: *ip.as_inner(),
                sin6_flowinfo: hton(flowinfo),
                sin6_scope_id: hton(scope_id),
                .. unsafe { mem::zeroed() }
            },
        }
    }

    /// Returns the IP address associated with this socket address.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn ip(&self) -> &Ipv6Addr {
        unsafe {
            &*(&self.inner.sin6_addr as *const c::in6_addr as *const Ipv6Addr)
        }
    }

    /// Returns the port number associated with this socket address.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn port(&self) -> u16 { ntoh(self.inner.sin6_port) }

    /// Returns scope ID associated with this address, corresponding to the
    /// `sin6_flowinfo` field in C.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn flowinfo(&self) -> u32 { ntoh(self.inner.sin6_flowinfo) }

    /// Returns scope ID associated with this address, corresponding to the
    /// `sin6_scope_id` field in C.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn scope_id(&self) -> u32 { ntoh(self.inner.sin6_scope_id) }
}

impl FromInner<c::sockaddr_in> for SocketAddrV4 {
    fn from_inner(addr: c::sockaddr_in) -> SocketAddrV4 {
        SocketAddrV4 { inner: addr }
    }
}

impl FromInner<c::sockaddr_in6> for SocketAddrV6 {
    fn from_inner(addr: c::sockaddr_in6) -> SocketAddrV6 {
        SocketAddrV6 { inner: addr }
    }
}

impl<'a> IntoInner<(*const c::sockaddr, c::socklen_t)> for &'a SocketAddr {
    fn into_inner(self) -> (*const c::sockaddr, c::socklen_t) {
        match *self {
            SocketAddr::V4(ref a) => {
                (a as *const _ as *const _, mem::size_of_val(a) as c::socklen_t)
            }
            SocketAddr::V6(ref a) => {
                (a as *const _ as *const _, mem::size_of_val(a) as c::socklen_t)
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for SocketAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SocketAddr::V4(ref a) => a.fmt(f),
            SocketAddr::V6(ref a) => a.fmt(f),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for SocketAddrV4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.ip(), self.port())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for SocketAddrV4 {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for SocketAddrV6 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}]:{}", self.ip(), self.port())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for SocketAddrV6 {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Clone for SocketAddrV4 {
    fn clone(&self) -> SocketAddrV4 { *self }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Clone for SocketAddrV6 {
    fn clone(&self) -> SocketAddrV6 { *self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for SocketAddrV4 {
    fn eq(&self, other: &SocketAddrV4) -> bool {
        self.inner.sin_port == other.inner.sin_port &&
            self.inner.sin_addr.s_addr == other.inner.sin_addr.s_addr
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for SocketAddrV6 {
    fn eq(&self, other: &SocketAddrV6) -> bool {
        self.inner.sin6_port == other.inner.sin6_port &&
            self.inner.sin6_addr.s6_addr == other.inner.sin6_addr.s6_addr &&
            self.inner.sin6_flowinfo == other.inner.sin6_flowinfo &&
            self.inner.sin6_scope_id == other.inner.sin6_scope_id
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for SocketAddrV4 {}
#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for SocketAddrV6 {}

#[stable(feature = "rust1", since = "1.0.0")]
impl hash::Hash for SocketAddrV4 {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        (self.inner.sin_port, self.inner.sin_addr.s_addr).hash(s)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl hash::Hash for SocketAddrV6 {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        (self.inner.sin6_port, &self.inner.sin6_addr.s6_addr,
         self.inner.sin6_flowinfo, self.inner.sin6_scope_id).hash(s)
    }
}

/// A trait for objects which can be converted or resolved to one or more
/// `SocketAddr` values.
///
/// This trait is used for generic address resolution when constructing network
/// objects.  By default it is implemented for the following types:
///
///  * `SocketAddr`, `SocketAddrV4`, `SocketAddrV6` - `to_socket_addrs` is
///    identity function.
///
///  * `(IpvNAddr, u16)` - `to_socket_addrs` constructs `SocketAddr` trivially.
///
///  * `(&str, u16)` - the string should be either a string representation of an
///    IP address expected by `FromStr` implementation for `IpvNAddr` or a host
///    name.
///
///  * `&str` - the string should be either a string representation of a
///    `SocketAddr` as expected by its `FromStr` implementation or a string like
///    `<host_name>:<port>` pair where `<port>` is a `u16` value.
///
/// This trait allows constructing network objects like `TcpStream` or
/// `UdpSocket` easily with values of various types for the bind/connection
/// address. It is needed because sometimes one type is more appropriate than
/// the other: for simple uses a string like `"localhost:12345"` is much nicer
/// than manual construction of the corresponding `SocketAddr`, but sometimes
/// `SocketAddr` value is *the* main source of the address, and converting it to
/// some other type (e.g. a string) just for it to be converted back to
/// `SocketAddr` in constructor methods is pointless.
///
/// Some examples:
///
/// ```no_run
/// use std::net::{SocketAddrV4, TcpStream, UdpSocket, TcpListener, Ipv4Addr};
///
/// fn main() {
///     let ip = Ipv4Addr::new(127, 0, 0, 1);
///     let port = 12345;
///
///     // The following lines are equivalent modulo possible "localhost" name
///     // resolution differences
///     let tcp_s = TcpStream::connect(SocketAddrV4::new(ip, port));
///     let tcp_s = TcpStream::connect((ip, port));
///     let tcp_s = TcpStream::connect(("127.0.0.1", port));
///     let tcp_s = TcpStream::connect(("localhost", port));
///     let tcp_s = TcpStream::connect("127.0.0.1:12345");
///     let tcp_s = TcpStream::connect("localhost:12345");
///
///     // TcpListener::bind(), UdpSocket::bind() and UdpSocket::send_to()
///     // behave similarly
///     let tcp_l = TcpListener::bind("localhost:12345");
///
///     let mut udp_s = UdpSocket::bind(("127.0.0.1", port)).unwrap();
///     udp_s.send_to(&[7], (ip, 23451)).unwrap();
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ToSocketAddrs {
    /// Returned iterator over socket addresses which this type may correspond
    /// to.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Iter: Iterator<Item=SocketAddr>;

    /// Converts this object to an iterator of resolved `SocketAddr`s.
    ///
    /// The returned iterator may not actually yield any values depending on the
    /// outcome of any resolution performed.
    ///
    /// Note that this function may block the current thread while resolution is
    /// performed.
    ///
    /// # Errors
    ///
    /// Any errors encountered during resolution will be returned as an `Err`.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn to_socket_addrs(&self) -> io::Result<Self::Iter>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToSocketAddrs for SocketAddr {
    type Iter = option::IntoIter<SocketAddr>;
    fn to_socket_addrs(&self) -> io::Result<option::IntoIter<SocketAddr>> {
        Ok(Some(*self).into_iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToSocketAddrs for SocketAddrV4 {
    type Iter = option::IntoIter<SocketAddr>;
    fn to_socket_addrs(&self) -> io::Result<option::IntoIter<SocketAddr>> {
        SocketAddr::V4(*self).to_socket_addrs()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToSocketAddrs for SocketAddrV6 {
    type Iter = option::IntoIter<SocketAddr>;
    fn to_socket_addrs(&self) -> io::Result<option::IntoIter<SocketAddr>> {
        SocketAddr::V6(*self).to_socket_addrs()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToSocketAddrs for (IpAddr, u16) {
    type Iter = option::IntoIter<SocketAddr>;
    fn to_socket_addrs(&self) -> io::Result<option::IntoIter<SocketAddr>> {
        let (ip, port) = *self;
        match ip {
            IpAddr::V4(ref a) => (*a, port).to_socket_addrs(),
            IpAddr::V6(ref a) => (*a, port).to_socket_addrs(),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToSocketAddrs for (Ipv4Addr, u16) {
    type Iter = option::IntoIter<SocketAddr>;
    fn to_socket_addrs(&self) -> io::Result<option::IntoIter<SocketAddr>> {
        let (ip, port) = *self;
        SocketAddrV4::new(ip, port).to_socket_addrs()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToSocketAddrs for (Ipv6Addr, u16) {
    type Iter = option::IntoIter<SocketAddr>;
    fn to_socket_addrs(&self) -> io::Result<option::IntoIter<SocketAddr>> {
        let (ip, port) = *self;
        SocketAddrV6::new(ip, port, 0, 0).to_socket_addrs()
    }
}

fn resolve_socket_addr(s: &str, p: u16) -> io::Result<vec::IntoIter<SocketAddr>> {
    let ips = try!(lookup_host(s));
    let v: Vec<_> = try!(ips.map(|a| {
        a.map(|a| {
            match a {
                SocketAddr::V4(ref a) => {
                    SocketAddr::V4(SocketAddrV4::new(*a.ip(), p))
                }
                SocketAddr::V6(ref a) => {
                    SocketAddr::V6(SocketAddrV6::new(*a.ip(), p, a.flowinfo(),
                                                     a.scope_id()))
                }
            }
        })
    }).collect());
    Ok(v.into_iter())
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> ToSocketAddrs for (&'a str, u16) {
    type Iter = vec::IntoIter<SocketAddr>;
    fn to_socket_addrs(&self) -> io::Result<vec::IntoIter<SocketAddr>> {
        let (host, port) = *self;

        // try to parse the host as a regular IP address first
        if let Ok(addr) = host.parse::<Ipv4Addr>() {
            let addr = SocketAddrV4::new(addr, port);
            return Ok(vec![SocketAddr::V4(addr)].into_iter())
        }
        if let Ok(addr) = host.parse::<Ipv6Addr>() {
            let addr = SocketAddrV6::new(addr, port, 0, 0);
            return Ok(vec![SocketAddr::V6(addr)].into_iter())
        }

        resolve_socket_addr(host, port)
    }
}

// accepts strings like 'localhost:12345'
#[stable(feature = "rust1", since = "1.0.0")]
impl ToSocketAddrs for str {
    type Iter = vec::IntoIter<SocketAddr>;
    fn to_socket_addrs(&self) -> io::Result<vec::IntoIter<SocketAddr>> {
        // try to parse as a regular SocketAddr first
        match self.parse().ok() {
            Some(addr) => return Ok(vec![addr].into_iter()),
            None => {}
        }

        macro_rules! try_opt {
            ($e:expr, $msg:expr) => (
                match $e {
                    Some(r) => r,
                    None => return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                                      $msg)),
                }
            )
        }

        // split the string by ':' and convert the second part to u16
        let mut parts_iter = self.rsplitn(2, ':');
        let port_str = try_opt!(parts_iter.next(), "invalid socket address");
        let host = try_opt!(parts_iter.next(), "invalid socket address");
        let port: u16 = try_opt!(port_str.parse().ok(), "invalid port value");
        resolve_socket_addr(host, port)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ToSocketAddrs + ?Sized> ToSocketAddrs for &'a T {
    type Iter = T::Iter;
    fn to_socket_addrs(&self) -> io::Result<T::Iter> {
        (**self).to_socket_addrs()
    }
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use io;
    use net::*;
    use net::Ipv6MulticastScope::*;
    use net::test::{tsa, sa6, sa4};

    #[test]
    fn to_socket_addr_ipaddr_u16() {
        let a = Ipv4Addr::new(77, 88, 21, 11);
        let p = 12345;
        let e = SocketAddr::V4(SocketAddrV4::new(a, p));
        assert_eq!(Ok(vec![e]), tsa((a, p)));
    }

    #[test]
    fn to_socket_addr_str_u16() {
        let a = sa4(Ipv4Addr::new(77, 88, 21, 11), 24352);
        assert_eq!(Ok(vec![a]), tsa(("77.88.21.11", 24352)));

        let a = sa6(Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 53);
        assert_eq!(Ok(vec![a]), tsa(("2a02:6b8:0:1::1", 53)));

        let a = sa4(Ipv4Addr::new(127, 0, 0, 1), 23924);
        assert!(tsa(("localhost", 23924)).unwrap().contains(&a));
    }

    #[test]
    fn to_socket_addr_str() {
        let a = sa4(Ipv4Addr::new(77, 88, 21, 11), 24352);
        assert_eq!(Ok(vec![a]), tsa("77.88.21.11:24352"));

        let a = sa6(Ipv6Addr::new(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 53);
        assert_eq!(Ok(vec![a]), tsa("[2a02:6b8:0:1::1]:53"));

        let a = sa4(Ipv4Addr::new(127, 0, 0, 1), 23924);
        assert!(tsa("localhost:23924").unwrap().contains(&a));
    }

    // FIXME: figure out why this fails on openbsd and bitrig and fix it
    #[test]
    #[cfg(not(any(windows, target_os = "openbsd", target_os = "bitrig")))]
    fn to_socket_addr_str_bad() {
        assert!(tsa("1200::AB00:1234::2552:7777:1313:34300").is_err());
    }
}
