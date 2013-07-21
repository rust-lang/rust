// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

mod rusti {
    #[abi = "rust-intrinsic"]
    extern "rust-intrinsic" {
        fn ctpop8(x: i8) -> i8;
        fn ctpop16(x: i16) -> i16;
        fn ctpop32(x: i32) -> i32;
        fn ctpop64(x: i64) -> i64;

        fn ctlz8(x: i8) -> i8;
        fn ctlz16(x: i16) -> i16;
        fn ctlz32(x: i32) -> i32;
        fn ctlz64(x: i64) -> i64;

        fn cttz8(x: i8) -> i8;
        fn cttz16(x: i16) -> i16;
        fn cttz32(x: i32) -> i32;
        fn cttz64(x: i64) -> i64;

        fn bswap16(x: i16) -> i16;
        fn bswap32(x: i32) -> i32;
        fn bswap64(x: i64) -> i64;
    }
}

pub fn main() {
    unsafe {
        use rusti::*;

        assert_eq!(ctpop8(0i8), 0i8);
        assert_eq!(ctpop16(0i16), 0i16);
        assert_eq!(ctpop32(0i32), 0i32);
        assert_eq!(ctpop64(0i64), 0i64);

        assert_eq!(ctpop8(1i8), 1i8);
        assert_eq!(ctpop16(1i16), 1i16);
        assert_eq!(ctpop32(1i32), 1i32);
        assert_eq!(ctpop64(1i64), 1i64);

        assert_eq!(ctpop8(10i8), 2i8);
        assert_eq!(ctpop16(10i16), 2i16);
        assert_eq!(ctpop32(10i32), 2i32);
        assert_eq!(ctpop64(10i64), 2i64);

        assert_eq!(ctpop8(100i8), 3i8);
        assert_eq!(ctpop16(100i16), 3i16);
        assert_eq!(ctpop32(100i32), 3i32);
        assert_eq!(ctpop64(100i64), 3i64);

        assert_eq!(ctpop8(-1i8), 8i8);
        assert_eq!(ctpop16(-1i16), 16i16);
        assert_eq!(ctpop32(-1i32), 32i32);
        assert_eq!(ctpop64(-1i64), 64i64);

        assert_eq!(ctlz8(0i8), 8i8);
        assert_eq!(ctlz16(0i16), 16i16);
        assert_eq!(ctlz32(0i32), 32i32);
        assert_eq!(ctlz64(0i64), 64i64);

        assert_eq!(ctlz8(1i8), 7i8);
        assert_eq!(ctlz16(1i16), 15i16);
        assert_eq!(ctlz32(1i32), 31i32);
        assert_eq!(ctlz64(1i64), 63i64);

        assert_eq!(ctlz8(10i8), 4i8);
        assert_eq!(ctlz16(10i16), 12i16);
        assert_eq!(ctlz32(10i32), 28i32);
        assert_eq!(ctlz64(10i64), 60i64);

        assert_eq!(ctlz8(100i8), 1i8);
        assert_eq!(ctlz16(100i16), 9i16);
        assert_eq!(ctlz32(100i32), 25i32);
        assert_eq!(ctlz64(100i64), 57i64);

        assert_eq!(cttz8(-1i8), 0i8);
        assert_eq!(cttz16(-1i16), 0i16);
        assert_eq!(cttz32(-1i32), 0i32);
        assert_eq!(cttz64(-1i64), 0i64);

        assert_eq!(cttz8(0i8), 8i8);
        assert_eq!(cttz16(0i16), 16i16);
        assert_eq!(cttz32(0i32), 32i32);
        assert_eq!(cttz64(0i64), 64i64);

        assert_eq!(cttz8(1i8), 0i8);
        assert_eq!(cttz16(1i16), 0i16);
        assert_eq!(cttz32(1i32), 0i32);
        assert_eq!(cttz64(1i64), 0i64);

        assert_eq!(cttz8(10i8), 1i8);
        assert_eq!(cttz16(10i16), 1i16);
        assert_eq!(cttz32(10i32), 1i32);
        assert_eq!(cttz64(10i64), 1i64);

        assert_eq!(cttz8(100i8), 2i8);
        assert_eq!(cttz16(100i16), 2i16);
        assert_eq!(cttz32(100i32), 2i32);
        assert_eq!(cttz64(100i64), 2i64);

        assert_eq!(cttz8(-1i8), 0i8);
        assert_eq!(cttz16(-1i16), 0i16);
        assert_eq!(cttz32(-1i32), 0i32);
        assert_eq!(cttz64(-1i64), 0i64);

        assert_eq!(bswap16(0x0A0Bi16), 0x0B0Ai16);
        assert_eq!(bswap32(0x0ABBCC0Di32), 0x0DCCBB0Ai32);
        assert_eq!(bswap64(0x0122334455667708i64), 0x0877665544332201i64);
    }
}
