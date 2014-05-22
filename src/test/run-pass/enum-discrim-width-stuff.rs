// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

macro_rules! check {
    ($m:ident, $t:ty, $v:expr) => {{
        mod $m {
            use std::mem::size_of;
            enum E {
                V = $v,
                A = 0
            }
            static C: E = V;
            pub fn check() {
                assert_eq!(size_of::<E>(), size_of::<$t>());
                assert_eq!(V as $t, $v);
                assert_eq!(C as $t, $v);
                assert_eq!(format!("{:?}", V), "V".to_strbuf());
                assert_eq!(format!("{:?}", C), "V".to_strbuf());
            }
        }
        $m::check();
    }}
}

pub fn main() {
    check!(a, u8, 0x17);
    check!(b, u8, 0xe8);
    check!(c, u16, 0x1727);
    check!(d, u16, 0xe8d8);
    check!(e, u32, 0x17273747);
    check!(f, u32, 0xe8d8c8b8);
    check!(g, u64, 0x1727374757677787u64);
    check!(h, u64, 0xe8d8c8b8a8988878u64);

    check!(z, i8, 0x17);
    check!(y, i8, -0x17);
    check!(x, i16, 0x1727);
    check!(w, i16, -0x1727);
    check!(v, i32, 0x17273747);
    check!(u, i32, -0x17273747);
    check!(t, i64, 0x1727374757677787);
    check!(s, i64, -0x1727374757677787);

    enum Simple { A, B }
    assert_eq!(::std::mem::size_of::<Simple>(), 1);
}
