// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C debug-assertions

// Check that we do *not* overflow on a number of edge cases.
// (compare with test/run-fail/overflowing-{lsh,rsh}*.rs)

fn main() {
    test_left_shift();
    test_right_shift();
}

fn test_left_shift() {
    // negative rhs can panic, but values in [0,N-1] are okay for iN

    macro_rules! tests {
        ($iN:ty, $uN:ty, $max_rhs:expr, $expect_i:expr, $expect_u:expr) => { {
            let x = (1 as $iN) << 0;
            assert_eq!(x, 1);
            let x = (1 as $uN) << 0;
            assert_eq!(x, 1);
            let x = (1 as $iN) << $max_rhs;
            assert_eq!(x, $expect_i);
            let x = (1 as $uN) << $max_rhs;
            assert_eq!(x, $expect_u);
            // high-order bits on LHS are silently discarded without panic.
            let x = (3 as $iN) << $max_rhs;
            assert_eq!(x, $expect_i);
            let x = (3 as $uN) << $max_rhs;
            assert_eq!(x, $expect_u);
        } }
    }

    let x = 1_i8 << 0;
    assert_eq!(x, 1);
    let x = 1_u8 << 0;
    assert_eq!(x, 1);
    let x = 1_i8 << 7;
    assert_eq!(x, std::i8::MIN);
    let x = 1_u8 << 7;
    assert_eq!(x, 0x80);
    // high-order bits on LHS are silently discarded without panic.
    let x = 3_i8 << 7;
    assert_eq!(x, std::i8::MIN);
    let x = 3_u8 << 7;
    assert_eq!(x, 0x80);

    // above is (approximately) expanded from:
    tests!(i8, u8, 7, std::i8::MIN, 0x80_u8);

    tests!(i16, u16, 15, std::i16::MIN, 0x8000_u16);
    tests!(i32, u32, 31, std::i32::MIN, 0x8000_0000_u32);
    tests!(i64, u64, 63, std::i64::MIN, 0x8000_0000_0000_0000_u64);
}

fn test_right_shift() {
    // negative rhs can panic, but values in [0,N-1] are okay for iN

    macro_rules! tests {
        ($iN:ty, $uN:ty, $max_rhs:expr,
         $signbit_i:expr, $highbit_i:expr, $highbit_u:expr) =>
        { {
            let x = (1 as $iN) >> 0;
            assert_eq!(x, 1);
            let x = (1 as $uN) >> 0;
            assert_eq!(x, 1);
            let x = ($highbit_i) >> $max_rhs-1;
            assert_eq!(x, 1);
            let x = ($highbit_u) >> $max_rhs;
            assert_eq!(x, 1);
            // sign-bit is carried by arithmetic right shift
            let x = ($signbit_i) >> $max_rhs;
            assert_eq!(x, -1);
            // low-order bits on LHS are silently discarded without panic.
            let x = ($highbit_i + 1) >> $max_rhs-1;
            assert_eq!(x, 1);
            let x = ($highbit_u + 1) >> $max_rhs;
            assert_eq!(x, 1);
            let x = ($signbit_i + 1) >> $max_rhs;
            assert_eq!(x, -1);
        } }
    }

    tests!(i8, u8, 7, std::i8::MIN, 0x40_i8, 0x80_u8);
    tests!(i16, u16, 15, std::i16::MIN, 0x4000_u16, 0x8000_u16);
    tests!(i32, u32, 31, std::i32::MIN, 0x4000_0000_u32, 0x8000_0000_u32);
    tests!(i64, u64, 63, std::i64::MIN,
           0x4000_0000_0000_0000_u64, 0x8000_0000_0000_0000_u64);
}
