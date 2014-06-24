// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(globs, intrinsics)]

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

        assert_eq!(ctpop8(0u8), 0u8);
        assert_eq!(ctpop16(0u16), 0u16);
        assert_eq!(ctpop32(0u32), 0u32);
        assert_eq!(ctpop64(0u64), 0u64);

        assert_eq!(ctpop8(1u8), 1u8);
        assert_eq!(ctpop16(1u16), 1u16);
        assert_eq!(ctpop32(1u32), 1u32);
        assert_eq!(ctpop64(1u64), 1u64);

        assert_eq!(ctpop8(10u8), 2u8);
        assert_eq!(ctpop16(10u16), 2u16);
        assert_eq!(ctpop32(10u32), 2u32);
        assert_eq!(ctpop64(10u64), 2u64);

        assert_eq!(ctpop8(100u8), 3u8);
        assert_eq!(ctpop16(100u16), 3u16);
        assert_eq!(ctpop32(100u32), 3u32);
        assert_eq!(ctpop64(100u64), 3u64);

        assert_eq!(ctpop8(-1u8), 8u8);
        assert_eq!(ctpop16(-1u16), 16u16);
        assert_eq!(ctpop32(-1u32), 32u32);
        assert_eq!(ctpop64(-1u64), 64u64);

        assert_eq!(ctlz8(0u8), 8u8);
        assert_eq!(ctlz16(0u16), 16u16);
        assert_eq!(ctlz32(0u32), 32u32);
        assert_eq!(ctlz64(0u64), 64u64);

        assert_eq!(ctlz8(1u8), 7u8);
        assert_eq!(ctlz16(1u16), 15u16);
        assert_eq!(ctlz32(1u32), 31u32);
        assert_eq!(ctlz64(1u64), 63u64);

        assert_eq!(ctlz8(10u8), 4u8);
        assert_eq!(ctlz16(10u16), 12u16);
        assert_eq!(ctlz32(10u32), 28u32);
        assert_eq!(ctlz64(10u64), 60u64);

        assert_eq!(ctlz8(100u8), 1u8);
        assert_eq!(ctlz16(100u16), 9u16);
        assert_eq!(ctlz32(100u32), 25u32);
        assert_eq!(ctlz64(100u64), 57u64);

        assert_eq!(cttz8(-1u8), 0u8);
        assert_eq!(cttz16(-1u16), 0u16);
        assert_eq!(cttz32(-1u32), 0u32);
        assert_eq!(cttz64(-1u64), 0u64);

        assert_eq!(cttz8(0u8), 8u8);
        assert_eq!(cttz16(0u16), 16u16);
        assert_eq!(cttz32(0u32), 32u32);
        assert_eq!(cttz64(0u64), 64u64);

        assert_eq!(cttz8(1u8), 0u8);
        assert_eq!(cttz16(1u16), 0u16);
        assert_eq!(cttz32(1u32), 0u32);
        assert_eq!(cttz64(1u64), 0u64);

        assert_eq!(cttz8(10u8), 1u8);
        assert_eq!(cttz16(10u16), 1u16);
        assert_eq!(cttz32(10u32), 1u32);
        assert_eq!(cttz64(10u64), 1u64);

        assert_eq!(cttz8(100u8), 2u8);
        assert_eq!(cttz16(100u16), 2u16);
        assert_eq!(cttz32(100u32), 2u32);
        assert_eq!(cttz64(100u64), 2u64);

        assert_eq!(cttz8(-1u8), 0u8);
        assert_eq!(cttz16(-1u16), 0u16);
        assert_eq!(cttz32(-1u32), 0u32);
        assert_eq!(cttz64(-1u64), 0u64);

        assert_eq!(bswap16(0x0A0Bu16), 0x0B0Au16);
        assert_eq!(bswap32(0x0ABBCC0Du32), 0x0DCCBB0Au32);
        assert_eq!(bswap64(0x0122334455667708u64), 0x0877665544332201u64);
    }
}
