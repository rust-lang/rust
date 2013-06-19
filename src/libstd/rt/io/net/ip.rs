// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::{Eq, TotalEq};

pub enum IpAddr {
    Ipv4(u8, u8, u8, u8, u16),
    Ipv6
}

impl Eq for IpAddr {
    fn eq(&self, other: &IpAddr) -> bool {
        match (*self, *other) {
            (Ipv4(a,b,c,d,e), Ipv4(f,g,h,i,j)) => (a,b,c,d,e) == (f,g,h,i,j),
            (Ipv6, Ipv6) => fail!(), 
            _ => false
        }
    }
    fn ne(&self, other: &IpAddr) -> bool {
        !(self == other)
    }
}

impl TotalEq for IpAddr {
    fn equals(&self, other: &IpAddr) -> bool {
        *self == *other
    }
}
