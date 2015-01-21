// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Internet Protocol (IP) addresses.
//!
//! This module contains functions useful for parsing, formatting, and
//! manipulating IP addresses.

#![allow(missing_docs)]

pub use self::IpAddr::*;

use boxed::Box;
use fmt;
use io::{self, IoResult, IoError};
use io::net;
use iter::{Iterator, IteratorExt};
use ops::{FnOnce, FnMut};
use option::Option;
use option::Option::{None, Some};
use result::Result::{Ok, Err};
use slice::SliceExt;
use str::{FromStr, StrExt};
use vec::Vec;

pub type Port = u16;

#[derive(Copy, PartialEq, Eq, Clone, Hash, Show)]
pub enum IpAddr {
    Ipv4Addr(u8, u8, u8, u8),
    Ipv6Addr(u16, u16, u16, u16, u16, u16, u16, u16)
}

#[stable]
impl fmt::Display for IpAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Ipv4Addr(a, b, c, d) =>
                write!(fmt, "{}.{}.{}.{}", a, b, c, d),

            // Ipv4 Compatible address
            Ipv6Addr(0, 0, 0, 0, 0, 0, g, h) => {
                write!(fmt, "::{}.{}.{}.{}", (g >> 8) as u8, g as u8,
                       (h >> 8) as u8, h as u8)
            }

            // Ipv4-Mapped address
            Ipv6Addr(0, 0, 0, 0, 0, 0xFFFF, g, h) => {
                write!(fmt, "::FFFF:{}.{}.{}.{}", (g >> 8) as u8, g as u8,
                       (h >> 8) as u8, h as u8)
            }

            Ipv6Addr(a, b, c, d, e, f, g, h) =>
                write!(fmt, "{:x}:{:x}:{:x}:{:x}:{:x}:{:x}:{:x}:{:x}",
                       a, b, c, d, e, f, g, h)
        }
    }
}

#[derive(Copy, PartialEq, Eq, Clone, Hash, Show)]
pub struct SocketAddr {
    pub ip: IpAddr,
    pub port: Port,
}

#[stable]
impl fmt::Display for SocketAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.ip {
            Ipv4Addr(..) => write!(f, "{}:{}", self.ip, self.port),
            Ipv6Addr(..) => write!(f, "[{}]:{}", self.ip, self.port),
        }
    }
}

struct Parser<'a> {
    // parsing as ASCII, so can use byte array
    s: &'a [u8],
    pos: uint,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Parser<'a> {
        Parser {
            s: s.as_bytes(),
            pos: 0,
        }
    }

    fn is_eof(&self) -> bool {
        self.pos == self.s.len()
    }

    // Commit only if parser returns Some
    fn read_atomically<T, F>(&mut self, cb: F) -> Option<T> where
        F: FnOnce(&mut Parser) -> Option<T>,
    {
        let pos = self.pos;
        let r = cb(self);
        if r.is_none() {
            self.pos = pos;
        }
        r
    }

    // Commit only if parser read till EOF
    fn read_till_eof<T, F>(&mut self, cb: F) -> Option<T> where
        F: FnOnce(&mut Parser) -> Option<T>,
    {
        self.read_atomically(move |p| {
            match cb(p) {
                Some(x) => if p.is_eof() {Some(x)} else {None},
                None => None,
            }
        })
    }

    // Return result of first successful parser
    fn read_or<T>(&mut self, parsers: &mut [Box<FnMut(&mut Parser) -> Option<T>>])
               -> Option<T> {
        for pf in parsers.iter_mut() {
            match self.read_atomically(|p: &mut Parser| pf.call_mut((p,))) {
                Some(r) => return Some(r),
                None => {}
            }
        }
        None
    }

    // Apply 3 parsers sequentially
    fn read_seq_3<A, B, C, PA, PB, PC>(&mut self,
                                       pa: PA,
                                       pb: PB,
                                       pc: PC)
                                       -> Option<(A, B, C)> where
        PA: FnOnce(&mut Parser) -> Option<A>,
        PB: FnOnce(&mut Parser) -> Option<B>,
        PC: FnOnce(&mut Parser) -> Option<C>,
    {
        self.read_atomically(move |p| {
            let a = pa(p);
            let b = if a.is_some() { pb(p) } else { None };
            let c = if b.is_some() { pc(p) } else { None };
            match (a, b, c) {
                (Some(a), Some(b), Some(c)) => Some((a, b, c)),
                _ => None
            }
        })
    }

    // Read next char
    fn read_char(&mut self) -> Option<char> {
        if self.is_eof() {
            None
        } else {
            let r = self.s[self.pos] as char;
            self.pos += 1;
            Some(r)
        }
    }

    // Return char and advance iff next char is equal to requested
    fn read_given_char(&mut self, c: char) -> Option<char> {
        self.read_atomically(|p| {
            match p.read_char() {
                Some(next) if next == c => Some(next),
                _ => None,
            }
        })
    }

    // Read digit
    fn read_digit(&mut self, radix: u8) -> Option<u8> {
        fn parse_digit(c: char, radix: u8) -> Option<u8> {
            let c = c as u8;
            // assuming radix is either 10 or 16
            if c >= b'0' && c <= b'9' {
                Some(c - b'0')
            } else if radix > 10 && c >= b'a' && c < b'a' + (radix - 10) {
                Some(c - b'a' + 10)
            } else if radix > 10 && c >= b'A' && c < b'A' + (radix - 10) {
                Some(c - b'A' + 10)
            } else {
                None
            }
        }

        self.read_atomically(|p| {
            p.read_char().and_then(|c| parse_digit(c, radix))
        })
    }

    fn read_number_impl(&mut self, radix: u8, max_digits: u32, upto: u32) -> Option<u32> {
        let mut r = 0u32;
        let mut digit_count = 0;
        loop {
            match self.read_digit(radix) {
                Some(d) => {
                    r = r * (radix as u32) + (d as u32);
                    digit_count += 1;
                    if digit_count > max_digits || r >= upto {
                        return None
                    }
                }
                None => {
                    if digit_count == 0 {
                        return None
                    } else {
                        return Some(r)
                    }
                }
            };
        }
    }

    // Read number, failing if max_digits of number value exceeded
    fn read_number(&mut self, radix: u8, max_digits: u32, upto: u32) -> Option<u32> {
        self.read_atomically(|p| p.read_number_impl(radix, max_digits, upto))
    }

    fn read_ipv4_addr_impl(&mut self) -> Option<IpAddr> {
        let mut bs = [0u8; 4];
        let mut i = 0;
        while i < 4 {
            if i != 0 && self.read_given_char('.').is_none() {
                return None;
            }

            let octet = self.read_number(10, 3, 0x100).map(|n| n as u8);
            match octet {
                Some(d) => bs[i] = d,
                None => return None,
            };
            i += 1;
        }
        Some(Ipv4Addr(bs[0], bs[1], bs[2], bs[3]))
    }

    // Read IPv4 address
    fn read_ipv4_addr(&mut self) -> Option<IpAddr> {
        self.read_atomically(|p| p.read_ipv4_addr_impl())
    }

    fn read_ipv6_addr_impl(&mut self) -> Option<IpAddr> {
        fn ipv6_addr_from_head_tail(head: &[u16], tail: &[u16]) -> IpAddr {
            assert!(head.len() + tail.len() <= 8);
            let mut gs = [0u16; 8];
            gs.clone_from_slice(head);
            gs[(8 - tail.len()) .. 8].clone_from_slice(tail);
            Ipv6Addr(gs[0], gs[1], gs[2], gs[3], gs[4], gs[5], gs[6], gs[7])
        }

        fn read_groups(p: &mut Parser, groups: &mut [u16; 8], limit: uint) -> (uint, bool) {
            let mut i = 0;
            while i < limit {
                if i < limit - 1 {
                    let ipv4 = p.read_atomically(|p| {
                        if i == 0 || p.read_given_char(':').is_some() {
                            p.read_ipv4_addr()
                        } else {
                            None
                        }
                    });
                    match ipv4 {
                        Some(Ipv4Addr(a, b, c, d)) => {
                            groups[i + 0] = ((a as u16) << 8) | (b as u16);
                            groups[i + 1] = ((c as u16) << 8) | (d as u16);
                            return (i + 2, true);
                        }
                        _ => {}
                    }
                }

                let group = p.read_atomically(|p| {
                    if i == 0 || p.read_given_char(':').is_some() {
                        p.read_number(16, 4, 0x10000).map(|n| n as u16)
                    } else {
                        None
                    }
                });
                match group {
                    Some(g) => groups[i] = g,
                    None => return (i, false)
                }
                i += 1;
            }
            (i, false)
        }

        let mut head = [0u16; 8];
        let (head_size, head_ipv4) = read_groups(self, &mut head, 8);

        if head_size == 8 {
            return Some(Ipv6Addr(
                head[0], head[1], head[2], head[3],
                head[4], head[5], head[6], head[7]))
        }

        // IPv4 part is not allowed before `::`
        if head_ipv4 {
            return None
        }

        // read `::` if previous code parsed less than 8 groups
        if !self.read_given_char(':').is_some() || !self.read_given_char(':').is_some() {
            return None;
        }

        let mut tail = [0u16; 8];
        let (tail_size, _) = read_groups(self, &mut tail, 8 - head_size);
        Some(ipv6_addr_from_head_tail(&head[..head_size], &tail[..tail_size]))
    }

    fn read_ipv6_addr(&mut self) -> Option<IpAddr> {
        self.read_atomically(|p| p.read_ipv6_addr_impl())
    }

    fn read_ip_addr(&mut self) -> Option<IpAddr> {
        let ipv4_addr = |&mut: p: &mut Parser| p.read_ipv4_addr();
        let ipv6_addr = |&mut: p: &mut Parser| p.read_ipv6_addr();
        self.read_or(&mut [box ipv4_addr, box ipv6_addr])
    }

    fn read_socket_addr(&mut self) -> Option<SocketAddr> {
        let ip_addr = |&: p: &mut Parser| {
            let ipv4_p = |&mut: p: &mut Parser| p.read_ip_addr();
            let ipv6_p = |&mut: p: &mut Parser| {
                let open_br = |&: p: &mut Parser| p.read_given_char('[');
                let ip_addr = |&: p: &mut Parser| p.read_ipv6_addr();
                let clos_br = |&: p: &mut Parser| p.read_given_char(']');
                p.read_seq_3::<char, IpAddr, char, _, _, _>(open_br, ip_addr, clos_br)
                        .map(|t| match t { (_, ip, _) => ip })
            };
            p.read_or(&mut [box ipv4_p, box ipv6_p])
        };
        let colon = |&: p: &mut Parser| p.read_given_char(':');
        let port  = |&: p: &mut Parser| p.read_number(10, 5, 0x10000).map(|n| n as u16);

        // host, colon, port
        self.read_seq_3::<IpAddr, char, u16, _, _, _>(ip_addr, colon, port)
                .map(|t| match t { (ip, _, port) => SocketAddr { ip: ip, port: port } })
    }
}

impl FromStr for IpAddr {
    fn from_str(s: &str) -> Option<IpAddr> {
        Parser::new(s).read_till_eof(|p| p.read_ip_addr())
    }
}

impl FromStr for SocketAddr {
    fn from_str(s: &str) -> Option<SocketAddr> {
        Parser::new(s).read_till_eof(|p| p.read_socket_addr())
    }
}

/// A trait for objects which can be converted or resolved to one or more `SocketAddr` values.
///
/// Implementing types minimally have to implement either `to_socket_addr` or `to_socket_addr_all`
/// method, and its trivial counterpart will be available automatically.
///
/// This trait is used for generic address resolution when constructing network objects.
/// By default it is implemented for the following types:
///
///  * `SocketAddr` - `to_socket_addr` is identity function.
///
///  * `(IpAddr, u16)` - `to_socket_addr` constructs `SocketAddr` trivially.
///
///  * `(&str, u16)` - the string should be either a string representation of an IP address
///    expected by `FromStr` implementation for `IpAddr` or a host name.
///
///    For the former, `to_socket_addr_all` returns a vector with a single element corresponding
///    to that IP address joined with the given port.
///
///    For the latter, it tries to resolve the host name and returns a vector of all IP addresses
///    for the host name, each joined with the given port.
///
///  * `&str` - the string should be either a string representation of a `SocketAddr` as
///    expected by its `FromStr` implementation or a string like `<host_name>:<port>` pair
///    where `<port>` is a `u16` value.
///
///    For the former, `to_socket_addr_all` returns a vector with a single element corresponding
///    to that socket address.
///
///    For the latter, it tries to resolve the host name and returns a vector of all IP addresses
///    for the host name, each joined with the port.
///
///
/// This trait allows constructing network objects like `TcpStream` or `UdpSocket` easily with
/// values of various types for the bind/connection address. It is needed because sometimes
/// one type is more appropriate than the other: for simple uses a string like `"localhost:12345"`
/// is much nicer than manual construction of the corresponding `SocketAddr`, but sometimes
/// `SocketAddr` value is *the* main source of the address, and converting it to some other type
/// (e.g. a string) just for it to be converted back to `SocketAddr` in constructor methods
/// is pointless.
///
/// Some examples:
///
/// ```rust,no_run
/// # #![allow(unused_must_use)]
///
/// use std::io::{TcpStream, TcpListener};
/// use std::io::net::udp::UdpSocket;
/// use std::io::net::ip::{Ipv4Addr, SocketAddr};
///
/// fn main() {
///     // The following lines are equivalent modulo possible "localhost" name resolution
///     // differences
///     let tcp_s = TcpStream::connect(SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: 12345 });
///     let tcp_s = TcpStream::connect((Ipv4Addr(127, 0, 0, 1), 12345u16));
///     let tcp_s = TcpStream::connect(("127.0.0.1", 12345u16));
///     let tcp_s = TcpStream::connect(("localhost", 12345u16));
///     let tcp_s = TcpStream::connect("127.0.0.1:12345");
///     let tcp_s = TcpStream::connect("localhost:12345");
///
///     // TcpListener::bind(), UdpSocket::bind() and UdpSocket::send_to() behave similarly
///     let tcp_l = TcpListener::bind("localhost:12345");
///
///     let mut udp_s = UdpSocket::bind(("127.0.0.1", 23451u16)).unwrap();
///     udp_s.send_to([7u8, 7u8, 7u8].as_slice(), (Ipv4Addr(127, 0, 0, 1), 23451u16));
/// }
/// ```
pub trait ToSocketAddr {
    /// Converts this object to single socket address value.
    ///
    /// If more than one value is available, this method returns the first one. If no
    /// values are available, this method returns an `IoError`.
    ///
    /// By default this method delegates to `to_socket_addr_all` method, taking the first
    /// item from its result.
    fn to_socket_addr(&self) -> IoResult<SocketAddr> {
        self.to_socket_addr_all()
            .and_then(|v| v.into_iter().next().ok_or_else(|| IoError {
                kind: io::InvalidInput,
                desc: "no address available",
                detail: None
            }))
    }

    /// Converts this object to all available socket address values.
    ///
    /// Some values like host name string naturally correspond to multiple IP addresses.
    /// This method tries to return all available addresses corresponding to this object.
    ///
    /// By default this method delegates to `to_socket_addr` method, creating a singleton
    /// vector from its result.
    #[inline]
    fn to_socket_addr_all(&self) -> IoResult<Vec<SocketAddr>> {
        self.to_socket_addr().map(|a| vec![a])
    }
}

impl ToSocketAddr for SocketAddr {
    #[inline]
    fn to_socket_addr(&self) -> IoResult<SocketAddr> { Ok(*self) }
}

impl ToSocketAddr for (IpAddr, u16) {
    #[inline]
    fn to_socket_addr(&self) -> IoResult<SocketAddr> {
        let (ip, port) = *self;
        Ok(SocketAddr { ip: ip, port: port })
    }
}

fn resolve_socket_addr(s: &str, p: u16) -> IoResult<Vec<SocketAddr>> {
    net::get_host_addresses(s)
        .map(|v| v.into_iter().map(|a| SocketAddr { ip: a, port: p }).collect())
}

fn parse_and_resolve_socket_addr(s: &str) -> IoResult<Vec<SocketAddr>> {
    macro_rules! try_opt {
        ($e:expr, $msg:expr) => (
            match $e {
                Some(r) => r,
                None => return Err(IoError {
                    kind: io::InvalidInput,
                    desc: $msg,
                    detail: None
                })
            }
        )
    }

    // split the string by ':' and convert the second part to u16
    let mut parts_iter = s.rsplitn(2, ':');
    let port_str = try_opt!(parts_iter.next(), "invalid socket address");
    let host = try_opt!(parts_iter.next(), "invalid socket address");
    let port: u16 = try_opt!(FromStr::from_str(port_str), "invalid port value");
    resolve_socket_addr(host, port)
}

impl<'a> ToSocketAddr for (&'a str, u16) {
    fn to_socket_addr_all(&self) -> IoResult<Vec<SocketAddr>> {
        let (host, port) = *self;

        // try to parse the host as a regular IpAddr first
        match FromStr::from_str(host) {
            Some(addr) => return Ok(vec![SocketAddr {
                ip: addr,
                port: port
            }]),
            None => {}
        }

        resolve_socket_addr(host, port)
    }
}

// accepts strings like 'localhost:12345'
impl<'a> ToSocketAddr for &'a str {
    fn to_socket_addr(&self) -> IoResult<SocketAddr> {
        // try to parse as a regular SocketAddr first
        match FromStr::from_str(*self) {
            Some(addr) => return Ok(addr),
            None => {}
        }

        parse_and_resolve_socket_addr(*self)
            .and_then(|v| v.into_iter().next()
                .ok_or_else(|| IoError {
                    kind: io::InvalidInput,
                    desc: "no address available",
                    detail: None
                })
            )
    }

    fn to_socket_addr_all(&self) -> IoResult<Vec<SocketAddr>> {
        // try to parse as a regular SocketAddr first
        match FromStr::from_str(*self) {
            Some(addr) => return Ok(vec![addr]),
            None => {}
        }

        parse_and_resolve_socket_addr(*self)
    }
}


#[cfg(test)]
mod test {
    use prelude::v1::*;
    use super::*;
    use str::FromStr;

    #[test]
    fn test_from_str_ipv4() {
        assert_eq!(Some(Ipv4Addr(127, 0, 0, 1)), FromStr::from_str("127.0.0.1"));
        assert_eq!(Some(Ipv4Addr(255, 255, 255, 255)), FromStr::from_str("255.255.255.255"));
        assert_eq!(Some(Ipv4Addr(0, 0, 0, 0)), FromStr::from_str("0.0.0.0"));

        // out of range
        let none: Option<IpAddr> = FromStr::from_str("256.0.0.1");
        assert_eq!(None, none);
        // too short
        let none: Option<IpAddr> = FromStr::from_str("255.0.0");
        assert_eq!(None, none);
        // too long
        let none: Option<IpAddr> = FromStr::from_str("255.0.0.1.2");
        assert_eq!(None, none);
        // no number between dots
        let none: Option<IpAddr> = FromStr::from_str("255.0..1");
        assert_eq!(None, none);
    }

    #[test]
    fn test_from_str_ipv6() {
        assert_eq!(Some(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 0)), FromStr::from_str("0:0:0:0:0:0:0:0"));
        assert_eq!(Some(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1)), FromStr::from_str("0:0:0:0:0:0:0:1"));

        assert_eq!(Some(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1)), FromStr::from_str("::1"));
        assert_eq!(Some(Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 0)), FromStr::from_str("::"));

        assert_eq!(Some(Ipv6Addr(0x2a02, 0x6b8, 0, 0, 0, 0, 0x11, 0x11)),
                FromStr::from_str("2a02:6b8::11:11"));

        // too long group
        let none: Option<IpAddr> = FromStr::from_str("::00000");
        assert_eq!(None, none);
        // too short
        let none: Option<IpAddr> = FromStr::from_str("1:2:3:4:5:6:7");
        assert_eq!(None, none);
        // too long
        let none: Option<IpAddr> = FromStr::from_str("1:2:3:4:5:6:7:8:9");
        assert_eq!(None, none);
        // triple colon
        let none: Option<IpAddr> = FromStr::from_str("1:2:::6:7:8");
        assert_eq!(None, none);
        // two double colons
        let none: Option<IpAddr> = FromStr::from_str("1:2::6::8");
        assert_eq!(None, none);
    }

    #[test]
    fn test_from_str_ipv4_in_ipv6() {
        assert_eq!(Some(Ipv6Addr(0, 0, 0, 0, 0, 0, 49152, 545)),
                FromStr::from_str("::192.0.2.33"));
        assert_eq!(Some(Ipv6Addr(0, 0, 0, 0, 0, 0xFFFF, 49152, 545)),
                FromStr::from_str("::FFFF:192.0.2.33"));
        assert_eq!(Some(Ipv6Addr(0x64, 0xff9b, 0, 0, 0, 0, 49152, 545)),
                FromStr::from_str("64:ff9b::192.0.2.33"));
        assert_eq!(Some(Ipv6Addr(0x2001, 0xdb8, 0x122, 0xc000, 0x2, 0x2100, 49152, 545)),
                FromStr::from_str("2001:db8:122:c000:2:2100:192.0.2.33"));

        // colon after v4
        let none: Option<IpAddr> = FromStr::from_str("::127.0.0.1:");
        assert_eq!(None, none);
        // not enough groups
        let none: Option<IpAddr> = FromStr::from_str("1.2.3.4.5:127.0.0.1");
        assert_eq!(None, none);
        // too many groups
        let none: Option<IpAddr> =
            FromStr::from_str("1.2.3.4.5:6:7:127.0.0.1");
        assert_eq!(None, none);
    }

    #[test]
    fn test_from_str_socket_addr() {
        assert_eq!(Some(SocketAddr { ip: Ipv4Addr(77, 88, 21, 11), port: 80 }),
                FromStr::from_str("77.88.21.11:80"));
        assert_eq!(Some(SocketAddr { ip: Ipv6Addr(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), port: 53 }),
                FromStr::from_str("[2a02:6b8:0:1::1]:53"));
        assert_eq!(Some(SocketAddr { ip: Ipv6Addr(0, 0, 0, 0, 0, 0, 0x7F00, 1), port: 22 }),
                FromStr::from_str("[::127.0.0.1]:22"));

        // without port
        let none: Option<SocketAddr> = FromStr::from_str("127.0.0.1");
        assert_eq!(None, none);
        // without port
        let none: Option<SocketAddr> = FromStr::from_str("127.0.0.1:");
        assert_eq!(None, none);
        // wrong brackets around v4
        let none: Option<SocketAddr> = FromStr::from_str("[127.0.0.1]:22");
        assert_eq!(None, none);
        // port out of range
        let none: Option<SocketAddr> = FromStr::from_str("127.0.0.1:123456");
        assert_eq!(None, none);
    }

    #[test]
    fn ipv6_addr_to_string() {
        let a1 = Ipv6Addr(0, 0, 0, 0, 0, 0xffff, 0xc000, 0x280);
        assert!(a1.to_string() == "::ffff:192.0.2.128" ||
                a1.to_string() == "::FFFF:192.0.2.128");
        assert_eq!(Ipv6Addr(8, 9, 10, 11, 12, 13, 14, 15).to_string(),
                   "8:9:a:b:c:d:e:f");
    }

    #[test]
    fn to_socket_addr_socketaddr() {
        let a = SocketAddr { ip: Ipv4Addr(77, 88, 21, 11), port: 12345 };
        assert_eq!(Ok(a), a.to_socket_addr());
        assert_eq!(Ok(vec![a]), a.to_socket_addr_all());
    }

    #[test]
    fn to_socket_addr_ipaddr_u16() {
        let a = Ipv4Addr(77, 88, 21, 11);
        let p = 12345u16;
        let e = SocketAddr { ip: a, port: p };
        assert_eq!(Ok(e), (a, p).to_socket_addr());
        assert_eq!(Ok(vec![e]), (a, p).to_socket_addr_all());
    }

    #[test]
    fn to_socket_addr_str_u16() {
        let a = SocketAddr { ip: Ipv4Addr(77, 88, 21, 11), port: 24352 };
        assert_eq!(Ok(a), ("77.88.21.11", 24352u16).to_socket_addr());
        assert_eq!(Ok(vec![a]), ("77.88.21.11", 24352u16).to_socket_addr_all());

        let a = SocketAddr { ip: Ipv6Addr(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), port: 53 };
        assert_eq!(Ok(a), ("2a02:6b8:0:1::1", 53).to_socket_addr());
        assert_eq!(Ok(vec![a]), ("2a02:6b8:0:1::1", 53).to_socket_addr_all());

        let a = SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: 23924 };
        assert!(("localhost", 23924u16).to_socket_addr_all().unwrap().contains(&a));
    }

    #[test]
    fn to_socket_addr_str() {
        let a = SocketAddr { ip: Ipv4Addr(77, 88, 21, 11), port: 24352 };
        assert_eq!(Ok(a), "77.88.21.11:24352".to_socket_addr());
        assert_eq!(Ok(vec![a]), "77.88.21.11:24352".to_socket_addr_all());

        let a = SocketAddr { ip: Ipv6Addr(0x2a02, 0x6b8, 0, 1, 0, 0, 0, 1), port: 53 };
        assert_eq!(Ok(a), "[2a02:6b8:0:1::1]:53".to_socket_addr());
        assert_eq!(Ok(vec![a]), "[2a02:6b8:0:1::1]:53".to_socket_addr_all());

        let a = SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: 23924 };
        assert!("localhost:23924".to_socket_addr_all().unwrap().contains(&a));
    }
}
