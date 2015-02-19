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
use libc::{self, socklen_t, sa_family_t};
use mem;
use net::{IpAddr, lookup_host, ntoh, hton};
use option;
use sys_common::{FromInner, AsInner, IntoInner};
use vec;

/// Representation of a socket address for networking applications
///
/// A socket address consists of at least an (ip, port) pair and may also
/// contain other information depending on the protocol.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct SocketAddr {
    repr: Repr,
}

#[derive(Copy)]
enum Repr {
    V4(libc::sockaddr_in),
    V6(libc::sockaddr_in6),
}

impl SocketAddr {
    /// Creates a new socket address from the (ip, port) pair.
    pub fn new(ip: IpAddr, port: u16) -> SocketAddr {
        let repr = match ip {
            IpAddr::V4(ref ip) => {
                Repr::V4(libc::sockaddr_in {
                    sin_family: libc::AF_INET as sa_family_t,
                    sin_port: hton(port),
                    sin_addr: *ip.as_inner(),
                    .. unsafe { mem::zeroed() }
                })
            }
            IpAddr::V6(ref ip) => {
                Repr::V6(libc::sockaddr_in6 {
                    sin6_family: libc::AF_INET6 as sa_family_t,
                    sin6_port: hton(port),
                    sin6_addr: *ip.as_inner(),
                    .. unsafe { mem::zeroed() }
                })
            }
        };
        SocketAddr { repr: repr }
    }

    /// Gets the IP address associated with this socket address.
    pub fn ip(&self) -> IpAddr {
        match self.repr {
            Repr::V4(ref sa) => IpAddr::V4(FromInner::from_inner(sa.sin_addr)),
            Repr::V6(ref sa) => IpAddr::V6(FromInner::from_inner(sa.sin6_addr)),
        }
    }

    /// Gets the port number associated with this socket address
    pub fn port(&self) -> u16 {
        match self.repr {
            Repr::V4(ref sa) => ntoh(sa.sin_port),
            Repr::V6(ref sa) => ntoh(sa.sin6_port),
        }
    }

    fn set_port(&mut self, port: u16) {
        match self.repr {
            Repr::V4(ref mut sa) => sa.sin_port = hton(port),
            Repr::V6(ref mut sa) => sa.sin6_port = hton(port),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for SocketAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.repr {
            Repr::V4(_) => write!(f, "{}:{}", self.ip(), self.port()),
            Repr::V6(_) => write!(f, "[{}]:{}", self.ip(), self.port()),
        }
    }
}

impl FromInner<libc::sockaddr_in> for SocketAddr {
    fn from_inner(addr: libc::sockaddr_in) -> SocketAddr {
        SocketAddr { repr: Repr::V4(addr) }
    }
}

impl FromInner<libc::sockaddr_in6> for SocketAddr {
    fn from_inner(addr: libc::sockaddr_in6) -> SocketAddr {
        SocketAddr { repr: Repr::V6(addr) }
    }
}

impl<'a> IntoInner<(*const libc::sockaddr, socklen_t)> for &'a SocketAddr {
    fn into_inner(self) -> (*const libc::sockaddr, socklen_t) {
        match self.repr {
            Repr::V4(ref a) => {
                (a as *const _ as *const _, mem::size_of_val(a) as socklen_t)
            }
            Repr::V6(ref a) => {
                (a as *const _ as *const _, mem::size_of_val(a) as socklen_t)
            }
        }
    }
}

impl fmt::Debug for SocketAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, fmt)
    }
}

impl Clone for Repr {
    fn clone(&self) -> Repr { *self }
}

impl PartialEq for Repr {
    fn eq(&self, other: &Repr) -> bool {
        match (*self, *other) {
            (Repr::V4(ref a), Repr::V4(ref b)) => {
                a.sin_port == b.sin_port &&
                    a.sin_addr.s_addr == b.sin_addr.s_addr
            }
            (Repr::V6(ref a), Repr::V6(ref b)) => {
                a.sin6_port == b.sin6_port &&
                    a.sin6_addr.s6_addr == b.sin6_addr.s6_addr &&
                    a.sin6_flowinfo == b.sin6_flowinfo &&
                    a.sin6_scope_id == b.sin6_scope_id
            }
            _ => false,
        }
    }
}
impl Eq for Repr {}

#[cfg(stage0)]
impl<S: hash::Hasher + hash::Writer> hash::Hash<S> for Repr {
    fn hash(&self, s: &mut S) {
        match *self {
            Repr::V4(ref a) => {
                (a.sin_family, a.sin_port, a.sin_addr.s_addr).hash(s)
            }
            Repr::V6(ref a) => {
                (a.sin6_family, a.sin6_port, &a.sin6_addr.s6_addr,
                 a.sin6_flowinfo, a.sin6_scope_id).hash(s)
            }
        }
    }
}
#[cfg(not(stage0))]
#[stable(feature = "rust1", since = "1.0.0")]
impl hash::Hash for Repr {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        match *self {
            Repr::V4(ref a) => {
                (a.sin_family, a.sin_port, a.sin_addr.s_addr).hash(s)
            }
            Repr::V6(ref a) => {
                (a.sin6_family, a.sin6_port, &a.sin6_addr.s6_addr,
                 a.sin6_flowinfo, a.sin6_scope_id).hash(s)
            }
        }
    }
}

/// A trait for objects which can be converted or resolved to one or more
/// `SocketAddr` values.
///
/// This trait is used for generic address resolution when constructing network
/// objects.  By default it is implemented for the following types:
///
///  * `SocketAddr` - `to_socket_addrs` is identity function.
///
///  * `(IpAddr, u16)` - `to_socket_addrs` constructs `SocketAddr` trivially.
///
///  * `(&str, u16)` - the string should be either a string representation of an
///    IP address expected by `FromStr` implementation for `IpAddr` or a host
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
/// use std::net::{IpAddr, SocketAddr, TcpStream, UdpSocket, TcpListener};
///
/// fn main() {
///     let ip = IpAddr::new_v4(127, 0, 0, 1);
///     let port = 12345;
///
///     // The following lines are equivalent modulo possible "localhost" name
///     // resolution differences
///     let tcp_s = TcpStream::connect(&SocketAddr::new(ip, port));
///     let tcp_s = TcpStream::connect(&(ip, port));
///     let tcp_s = TcpStream::connect(&("127.0.0.1", port));
///     let tcp_s = TcpStream::connect(&("localhost", port));
///     let tcp_s = TcpStream::connect("127.0.0.1:12345");
///     let tcp_s = TcpStream::connect("localhost:12345");
///
///     // TcpListener::bind(), UdpSocket::bind() and UdpSocket::send_to()
///     // behave similarly
///     let tcp_l = TcpListener::bind("localhost:12345");
///
///     let mut udp_s = UdpSocket::bind(&("127.0.0.1", port)).unwrap();
///     udp_s.send_to(&[7], &(ip, 23451));
/// }
/// ```
pub trait ToSocketAddrs {
    /// Returned iterator over socket addresses which this type may correspond
    /// to.
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
    fn to_socket_addrs(&self) -> io::Result<Self::Iter>;
}

impl ToSocketAddrs for SocketAddr {
    type Iter = option::IntoIter<SocketAddr>;
    fn to_socket_addrs(&self) -> io::Result<option::IntoIter<SocketAddr>> {
        Ok(Some(*self).into_iter())
    }
}

impl ToSocketAddrs for (IpAddr, u16) {
    type Iter = option::IntoIter<SocketAddr>;
    fn to_socket_addrs(&self) -> io::Result<option::IntoIter<SocketAddr>> {
        let (ip, port) = *self;
        Ok(Some(SocketAddr::new(ip, port)).into_iter())
    }
}

fn resolve_socket_addr(s: &str, p: u16) -> io::Result<vec::IntoIter<SocketAddr>> {
    let ips = try!(lookup_host(s));
    let v: Vec<_> = try!(ips.map(|a| {
        a.map(|mut a| { a.set_port(p); a })
    }).collect());
    Ok(v.into_iter())
}

impl<'a> ToSocketAddrs for (&'a str, u16) {
    type Iter = vec::IntoIter<SocketAddr>;
    fn to_socket_addrs(&self) -> io::Result<vec::IntoIter<SocketAddr>> {
        let (host, port) = *self;

        // try to parse the host as a regular IpAddr first
        match host.parse().ok() {
            Some(addr) => return Ok(vec![SocketAddr::new(addr, port)].into_iter()),
            None => {}
        }

        resolve_socket_addr(host, port)
    }
}

// accepts strings like 'localhost:12345'
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
                                                      $msg, None)),
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

    #[test]
    fn test_from_str_ipv4() {
        assert_eq!(Ok(Ipv4Addr::new(127, 0, 0, 1)), "127.0.0.1".parse());
        assert_eq!(Ok(Ipv4Addr::new(255, 255, 255, 255)), "255.255.255.255".parse());
        assert_eq!(Ok(Ipv4Addr::new(0, 0, 0, 0)), "0.0.0.0".parse());

        // out of range
        let none: Option<IpAddr> = "256.0.0.1".parse().ok();
        assert_eq!(None, none);
        // too short
        let none: Option<IpAddr> = "255.0.0".parse().ok();
        assert_eq!(None, none);
        // too long
        let none: Option<IpAddr> = "255.0.0.1.2".parse().ok();
        assert_eq!(None, none);
        // no number between dots
        let none: Option<IpAddr> = "255.0..1".parse().ok();
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
        let none: Option<IpAddr> = "::00000".parse().ok();
        assert_eq!(None, none);
        // too short
        let none: Option<IpAddr> = "1:2:3:4:5:6:7".parse().ok();
        assert_eq!(None, none);
        // too long
        let none: Option<IpAddr> = "1:2:3:4:5:6:7:8:9".parse().ok();
        assert_eq!(None, none);
        // triple colon
        let none: Option<IpAddr> = "1:2:::6:7:8".parse().ok();
        assert_eq!(None, none);
        // two double colons
        let none: Option<IpAddr> = "1:2::6::8".parse().ok();
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
        let none: Option<IpAddr> = "::127.0.0.1:".parse().ok();
        assert_eq!(None, none);
        // not enough groups
        let none: Option<IpAddr> = "1.2.3.4.5:127.0.0.1".parse().ok();
        assert_eq!(None, none);
        // too many groups
        let none: Option<IpAddr> = "1.2.3.4.5:6:7:127.0.0.1".parse().ok();
        assert_eq!(None, none);
    }

    #[test]
    fn test_from_str_socket_addr() {
        assert_eq!(Ok(SocketAddr::new(IpAddr::new_v4(77, 88, 21, 11), 80)),
                "77.88.21.11:80".parse());
        assert_eq!(Ok(SocketAddr::new(IpAddr::new_v6(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 53)),
                "[2a02:6b8:0:1::1]:53".parse());
        assert_eq!(Ok(SocketAddr::new(IpAddr::new_v6(0, 0, 0, 0, 0, 0, 0x7F00, 1), 22)),
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
                 multicast: bool) {
            let ip = Ipv4Addr::new(octets[0], octets[1], octets[2], octets[3]);
            assert_eq!(octets, &ip.octets());

            assert_eq!(ip.is_unspecified(), unspec);
            assert_eq!(ip.is_loopback(), loopback);
            assert_eq!(ip.is_private(), private);
            assert_eq!(ip.is_link_local(), link_local);
            assert_eq!(ip.is_global(), global);
            assert_eq!(ip.is_multicast(), multicast);
        }

        //    address                unspec loopbk privt  linloc global multicast
        check(&[0, 0, 0, 0],         true,  false, false, false, true,  false);
        check(&[0, 0, 0, 1],         false, false, false, false, true,  false);
        check(&[1, 0, 0, 0],         false, false, false, false, true,  false);
        check(&[10, 9, 8, 7],        false, false, true,  false, false, false);
        check(&[127, 1, 2, 3],       false, true,  false, false, false, false);
        check(&[172, 31, 254, 253],  false, false, true,  false, false,  false);
        check(&[169, 254, 253, 242], false, false, false, true,  false, false);
        check(&[192, 168, 254, 253], false, false, true,  false, false, false);
        check(&[224, 0, 0, 0],       false, false, false, false, true,  true);
        check(&[239, 255, 255, 255], false, false, false, false, true,  true);
        check(&[255, 255, 255, 255], false, false, false, false, true,  false);
    }

    #[test]
    fn ipv6_properties() {
        fn check(str_addr: &str, unspec: bool, loopback: bool,
                 unique_local: bool, global: bool,
                 u_link_local: bool, u_site_local: bool, u_global: bool,
                 m_scope: Option<Ipv6MulticastScope>) {
            let ip: Ipv6Addr = str_addr.parse().ok().unwrap();
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

    fn tsa<A: ToSocketAddrs>(a: A) -> io::Result<Vec<SocketAddr>> {
        Ok(try!(a.to_socket_addrs()).collect())
    }

    #[test]
    fn to_socket_addr_socketaddr() {
        let a = SocketAddr::new(IpAddr::new_v4(77, 88, 21, 11), 12345);
        assert_eq!(Ok(vec![a]), tsa(a));
    }

    #[test]
    fn to_socket_addr_ipaddr_u16() {
        let a = IpAddr::new_v4(77, 88, 21, 11);
        let p = 12345u16;
        let e = SocketAddr::new(a, p);
        assert_eq!(Ok(vec![e]), tsa((a, p)));
    }

    #[test]
    fn to_socket_addr_str_u16() {
        let a = SocketAddr::new(IpAddr::new_v4(77, 88, 21, 11), 24352);
        assert_eq!(Ok(vec![a]), tsa(("77.88.21.11", 24352u16)));

        let a = SocketAddr::new(IpAddr::new_v6(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 53);
        assert_eq!(Ok(vec![a]), tsa(("2a02:6b8:0:1::1", 53)));

        let a = SocketAddr::new(IpAddr::new_v4(127, 0, 0, 1), 23924);
        assert!(tsa(("localhost", 23924u16)).unwrap().contains(&a));
    }

    #[test]
    fn to_socket_addr_str() {
        let a = SocketAddr::new(IpAddr::new_v4(77, 88, 21, 11), 24352);
        assert_eq!(Ok(vec![a]), tsa("77.88.21.11:24352"));

        let a = SocketAddr::new(IpAddr::new_v6(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), 53);
        assert_eq!(Ok(vec![a]), tsa("[2a02:6b8:0:1::1]:53"));

        let a = SocketAddr::new(IpAddr::new_v4(127, 0, 0, 1), 23924);
        assert!(tsa("localhost:23924").unwrap().contains(&a));
    }
}
