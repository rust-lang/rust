// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use container::Container;
use fmt;
use from_str::FromStr;
use iter::Iterator;
use option::{Option, None, Some};
use str::StrSlice;
use vec::{MutableCloneableVector, ImmutableVector, MutableVector};

pub type Port = u16;

#[deriving(Eq, TotalEq, Clone, Hash)]
pub enum IpAddr {
    Ipv4Addr(u8, u8, u8, u8),
    Ipv6Addr(u16, u16, u16, u16, u16, u16, u16, u16)
}

impl fmt::Show for IpAddr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Ipv4Addr(a, b, c, d) =>
                write!(fmt.buf, "{}.{}.{}.{}", a, b, c, d),

            // Ipv4 Compatible address
            Ipv6Addr(0, 0, 0, 0, 0, 0, g, h) => {
                write!(fmt.buf, "::{}.{}.{}.{}", (g >> 8) as u8, g as u8,
                       (h >> 8) as u8, h as u8)
            }

            // Ipv4-Mapped address
            Ipv6Addr(0, 0, 0, 0, 0, 0xFFFF, g, h) => {
                write!(fmt.buf, "::FFFF:{}.{}.{}.{}", (g >> 8) as u8, g as u8,
                       (h >> 8) as u8, h as u8)
            }

            Ipv6Addr(a, b, c, d, e, f, g, h) =>
                write!(fmt.buf, "{:x}:{:x}:{:x}:{:x}:{:x}:{:x}:{:x}:{:x}",
                       a, b, c, d, e, f, g, h)
        }
    }
}

#[deriving(Eq, TotalEq, Clone, Hash)]
pub struct SocketAddr {
    ip: IpAddr,
    port: Port,
}


impl fmt::Show for SocketAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.ip {
            Ipv4Addr(..) => write!(f.buf, "{}:{}", self.ip, self.port),
            Ipv6Addr(..) => write!(f.buf, "[{}]:{}", self.ip, self.port),
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
    fn read_atomically<T>(&mut self, cb: |&mut Parser| -> Option<T>)
                       -> Option<T> {
        let pos = self.pos;
        let r = cb(self);
        if r.is_none() {
            self.pos = pos;
        }
        r
    }

    // Commit only if parser read till EOF
    fn read_till_eof<T>(&mut self, cb: |&mut Parser| -> Option<T>)
                     -> Option<T> {
        self.read_atomically(|p| cb(p).filtered(|_| p.is_eof()))
    }

    // Return result of first successful parser
    fn read_or<T>(&mut self, parsers: &[|&mut Parser| -> Option<T>])
               -> Option<T> {
        for pf in parsers.iter() {
            match self.read_atomically(|p: &mut Parser| (*pf)(p)) {
                Some(r) => return Some(r),
                None => {}
            }
        }
        None
    }

    // Apply 3 parsers sequentially
    fn read_seq_3<A,
                  B,
                  C>(
                  &mut self,
                  pa: |&mut Parser| -> Option<A>,
                  pb: |&mut Parser| -> Option<B>,
                  pc: |&mut Parser| -> Option<C>)
                  -> Option<(A, B, C)> {
        self.read_atomically(|p| {
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
            p.read_char().filtered(|&next| next == c)
        })
    }

    // Read digit
    fn read_digit(&mut self, radix: u8) -> Option<u8> {
        fn parse_digit(c: char, radix: u8) -> Option<u8> {
            let c = c as u8;
            // assuming radix is either 10 or 16
            if c >= '0' as u8 && c <= '9' as u8 {
                Some(c - '0' as u8)
            } else if radix > 10 && c >= 'a' as u8 && c < 'a' as u8 + (radix - 10) {
                Some(c - 'a' as u8 + 10)
            } else if radix > 10 && c >= 'A' as u8 && c < 'A' as u8 + (radix - 10) {
                Some(c - 'A' as u8 + 10)
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
        let mut bs = [0u8, ..4];
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
            let mut gs = [0u16, ..8];
            gs.copy_from(head);
            gs.mut_slice(8 - tail.len(), 8).copy_from(tail);
            Ipv6Addr(gs[0], gs[1], gs[2], gs[3], gs[4], gs[5], gs[6], gs[7])
        }

        fn read_groups(p: &mut Parser, groups: &mut [u16, ..8], limit: uint) -> (uint, bool) {
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
                            groups[i + 0] = (a as u16 << 8) | (b as u16);
                            groups[i + 1] = (c as u16 << 8) | (d as u16);
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

        let mut head = [0u16, ..8];
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

        let mut tail = [0u16, ..8];
        let (tail_size, _) = read_groups(self, &mut tail, 8 - head_size);
        Some(ipv6_addr_from_head_tail(head.slice(0, head_size), tail.slice(0, tail_size)))
    }

    fn read_ipv6_addr(&mut self) -> Option<IpAddr> {
        self.read_atomically(|p| p.read_ipv6_addr_impl())
    }

    fn read_ip_addr(&mut self) -> Option<IpAddr> {
        let ipv4_addr = |p: &mut Parser| p.read_ipv4_addr();
        let ipv6_addr = |p: &mut Parser| p.read_ipv6_addr();
        self.read_or([ipv4_addr, ipv6_addr])
    }

    fn read_socket_addr(&mut self) -> Option<SocketAddr> {
        let ip_addr = |p: &mut Parser| {
            let ipv4_p = |p: &mut Parser| p.read_ip_addr();
            let ipv6_p = |p: &mut Parser| {
                let open_br = |p: &mut Parser| p.read_given_char('[');
                let ip_addr = |p: &mut Parser| p.read_ipv6_addr();
                let clos_br = |p: &mut Parser| p.read_given_char(']');
                p.read_seq_3::<char, IpAddr, char>(open_br, ip_addr, clos_br)
                        .map(|t| match t { (_, ip, _) => ip })
            };
            p.read_or([ipv4_p, ipv6_p])
        };
        let colon = |p: &mut Parser| p.read_given_char(':');
        let port  = |p: &mut Parser| p.read_number(10, 5, 0x10000).map(|n| n as u16);

        // host, colon, port
        self.read_seq_3::<IpAddr, char, u16>(ip_addr, colon, port)
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


#[cfg(test)]
mod test {
    use prelude::*;
    use super::*;
    use from_str::FromStr;

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
        // not enought groups
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
    fn ipv6_addr_to_str() {
        let a1 = Ipv6Addr(0, 0, 0, 0, 0, 0xffff, 0xc000, 0x280);
        assert!(a1.to_str() == ~"::ffff:192.0.2.128" || a1.to_str() == ~"::FFFF:192.0.2.128");
        assert_eq!(Ipv6Addr(8, 9, 10, 11, 12, 13, 14, 15).to_str(), ~"8:9:a:b:c:d:e:f");
    }
}
