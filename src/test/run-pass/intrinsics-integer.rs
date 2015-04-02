// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![feature(negate_unsigned)]
#![feature(intrinsics)]

mod rusti {
    extern "rust-intrinsic" {
        pub fn ctpop8(x: u8) -> u8;
        pub fn ctpop16(x: u16) -> u16;
        pub fn ctpop32(x: u32) -> u32;
        pub fn ctpop64(x: u64) -> u64;

        pub fn ctlz8(x: u8) -> u8;
        pub fn ctlz16(x: u16) -> u16;
        pub fn ctlz32(x: u32) -> u32;
        pub fn ctlz64(x: u64) -> u64;

        pub fn cttz8(x: u8) -> u8;
        pub fn cttz16(x: u16) -> u16;
        pub fn cttz32(x: u32) -> u32;
        pub fn cttz64(x: u64) -> u64;

        pub fn bswap16(x: u16) -> u16;
        pub fn bswap32(x: u32) -> u32;
        pub fn bswap64(x: u64) -> u64;
    }
}

pub fn main() {
    unsafe {
        use rusti::*;

        assert_eq!(ctpop8(0), 0);
        assert_eq!(ctpop16(0), 0);
        assert_eq!(ctpop32(0), 0);
        assert_eq!(ctpop64(0), 0);

        assert_eq!(ctpop8(1), 1);
        assert_eq!(ctpop16(1), 1);
        assert_eq!(ctpop32(1), 1);
        assert_eq!(ctpop64(1), 1);

        assert_eq!(ctpop8(10), 2);
        assert_eq!(ctpop16(10), 2);
        assert_eq!(ctpop32(10), 2);
        assert_eq!(ctpop64(10), 2);

        assert_eq!(ctpop8(100), 3);
        assert_eq!(ctpop16(100), 3);
        assert_eq!(ctpop32(100), 3);
        assert_eq!(ctpop64(100), 3);

        assert_eq!(ctpop8(-1), 8);
        assert_eq!(ctpop16(-1), 16);
        assert_eq!(ctpop32(-1), 32);
        assert_eq!(ctpop64(-1), 64);

        assert_eq!(ctlz8(0), 8);
        assert_eq!(ctlz16(0), 16);
        assert_eq!(ctlz32(0), 32);
        assert_eq!(ctlz64(0), 64);

        assert_eq!(ctlz8(1), 7);
        assert_eq!(ctlz16(1), 15);
        assert_eq!(ctlz32(1), 31);
        assert_eq!(ctlz64(1), 63);

        assert_eq!(ctlz8(10), 4);
        assert_eq!(ctlz16(10), 12);
        assert_eq!(ctlz32(10), 28);
        assert_eq!(ctlz64(10), 60);

        assert_eq!(ctlz8(100), 1);
        assert_eq!(ctlz16(100), 9);
        assert_eq!(ctlz32(100), 25);
        assert_eq!(ctlz64(100), 57);

        assert_eq!(cttz8(-1), 0);
        assert_eq!(cttz16(-1), 0);
        assert_eq!(cttz32(-1), 0);
        assert_eq!(cttz64(-1), 0);

        assert_eq!(cttz8(0), 8);
        assert_eq!(cttz16(0), 16);
        assert_eq!(cttz32(0), 32);
        assert_eq!(cttz64(0), 64);

        assert_eq!(cttz8(1), 0);
        assert_eq!(cttz16(1), 0);
        assert_eq!(cttz32(1), 0);
        assert_eq!(cttz64(1), 0);

        assert_eq!(cttz8(10), 1);
        assert_eq!(cttz16(10), 1);
        assert_eq!(cttz32(10), 1);
        assert_eq!(cttz64(10), 1);

        assert_eq!(cttz8(100), 2);
        assert_eq!(cttz16(100), 2);
        assert_eq!(cttz32(100), 2);
        assert_eq!(cttz64(100), 2);

        assert_eq!(cttz8(-1), 0);
        assert_eq!(cttz16(-1), 0);
        assert_eq!(cttz32(-1), 0);
        assert_eq!(cttz64(-1), 0);

        assert_eq!(bswap16(0x0A0B), 0x0B0A);
        assert_eq!(bswap32(0x0ABBCC0D), 0x0DCCBB0A);
        assert_eq!(bswap64(0x0122334455667708), 0x0877665544332201);
    }
}
