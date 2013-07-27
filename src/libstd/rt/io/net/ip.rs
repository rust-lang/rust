// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use num::FromStrRadix;
use to_str::ToStr;

type Port = u16;

#[deriving(Eq, TotalEq)]
pub enum IpAddr {
    Ipv4(u8, u8, u8, u8, Port),
    Ipv6(u16, u16, u16, u16, u16, u16, u16, u16, Port)
}

impl ToStr for IpAddr {
    fn to_str(&self) -> ~str {
        match *self {
            Ipv4(a, b, c, d, p) =>
                fmt!("%u.%u.%u.%u:%u",
                    a as uint, b as uint, c as uint, d as uint, p as uint),

            // Ipv4 Compatible address
            Ipv6(0, 0, 0, 0, 0, 0, g, h, p) => {
                let a = fmt!("%04x", g as uint);
                let b = FromStrRadix::from_str_radix(a.slice(2, 4), 16).unwrap();
                let a = FromStrRadix::from_str_radix(a.slice(0, 2), 16).unwrap();
                let c = fmt!("%04x", h as uint);
                let d = FromStrRadix::from_str_radix(c.slice(2, 4), 16).unwrap();
                let c = FromStrRadix::from_str_radix(c.slice(0, 2), 16).unwrap();

                fmt!("[::%u.%u.%u.%u]:%u", a, b, c, d, p as uint)
            }

            // Ipv4-Mapped address
            Ipv6(0, 0, 0, 0, 0, 1, g, h, p) => {
                let a = fmt!("%04x", g as uint);
                let b = FromStrRadix::from_str_radix(a.slice(2, 4), 16).unwrap();
                let a = FromStrRadix::from_str_radix(a.slice(0, 2), 16).unwrap();
                let c = fmt!("%04x", h as uint);
                let d = FromStrRadix::from_str_radix(c.slice(2, 4), 16).unwrap();
                let c = FromStrRadix::from_str_radix(c.slice(0, 2), 16).unwrap();

                fmt!("[::FFFF:%u.%u.%u.%u]:%u", a, b, c, d, p as uint)
            }

            Ipv6(a, b, c, d, e, f, g, h, p) =>
                fmt!("[%x:%x:%x:%x:%x:%x:%x:%x]:%u",
                    a as uint, b as uint, c as uint, d as uint,
                    e as uint, f as uint, g as uint, h as uint, p as uint)
        }
    }
}
