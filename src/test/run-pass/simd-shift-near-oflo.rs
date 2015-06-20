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

#![feature(core_simd)]

use std::simd::{i8x16, i16x8, i32x4, i64x2, u8x16, u16x8, u32x4, u64x2};

// (Work around constant-evaluation)
fn id<T>(x: T) -> T { x }

fn single_i8x16(x: i8) -> i8x16 { i8x16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x) }
fn single_u8x16(x: u8) -> u8x16 { u8x16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x) }
fn single_i16x8(x: i16) -> i16x8 { i16x8(0, 0, 0, 0, 0, 0, 0, x) }
fn single_u16x8(x: u16) -> u16x8 { u16x8(0, 0, 0, 0, 0, 0, 0, x) }
fn single_i32x4(x: i32) -> i32x4 { i32x4(0, 0, 0, x) }
fn single_u32x4(x: u32) -> u32x4 { u32x4(0, 0, 0, x) }
fn single_i64x2(x: i64) -> i64x2 { i64x2(0, x) }
fn single_u64x2(x: u64) -> u64x2 { u64x2(0, x) }

fn eq_i8x16(i8x16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15): i8x16,
            i8x16(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15): i8x16)
            -> bool {
    (x0 == y0) && (x1 == y1) && (x2 == y2) && (x3 == y3)
        && (x4 == y4) && (x5 == y5) && (x6 == y6) && (x7 == y7)
        && (x8 == y8) && (x9 == y9) && (x10 == y10) && (x11 == y11)
        && (x12 == y12) && (x13 == y13) && (x14 == y14) && (x15 == y15)
}
fn eq_u8x16(u8x16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15): u8x16,
            u8x16(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15): u8x16)
            -> bool {
    (x0 == y0) && (x1 == y1) && (x2 == y2) && (x3 == y3)
        && (x4 == y4) && (x5 == y5) && (x6 == y6) && (x7 == y7)
        && (x8 == y8) && (x9 == y9) && (x10 == y10) && (x11 == y11)
        && (x12 == y12) && (x13 == y13) && (x14 == y14) && (x15 == y15)
}
fn eq_i16x8(i16x8(x0, x1, x2, x3, x4, x5, x6, x7): i16x8,
            i16x8(y0, y1, y2, y3, y4, y5, y6, y7): i16x8) -> bool {
    (x0 == y0) && (x1 == y1) && (x2 == y2) && (x3 == y3)
        && (x4 == y4) && (x5 == y5) && (x6 == y6) && (x7 == y7)
}
fn eq_u16x8(u16x8(x0, x1, x2, x3, x4, x5, x6, x7): u16x8,
            u16x8(y0, y1, y2, y3, y4, y5, y6, y7): u16x8) -> bool {
    (x0 == y0) && (x1 == y1) && (x2 == y2) && (x3 == y3)
        && (x4 == y4) && (x5 == y5) && (x6 == y6) && (x7 == y7)
}
fn eq_i32x4(i32x4(x0, x1, x2, x3): i32x4, i32x4(y0, y1, y2, y3): i32x4) -> bool {
    (x0 == y0) && (x1 == y1) && (x2 == y2) && (x3 == y3)
}
fn eq_u32x4(u32x4(x0, x1, x2, x3): u32x4, u32x4(y0, y1, y2, y3): u32x4) -> bool {
    (x0 == y0) && (x1 == y1) && (x2 == y2) && (x3 == y3)
}
fn eq_i64x2(i64x2(x0, x1): i64x2, i64x2(y0, y1): i64x2) -> bool {
    (x0 == y0) && (x1 == y1)
}
fn eq_u64x2(u64x2(x0, x1): u64x2, u64x2(y0, y1): u64x2) -> bool {
    (x0 == y0) && (x1 == y1)
}

fn main() {
    test_left_shift();
    test_right_shift();
}

fn test_left_shift() {
    // negative rhs can panic, but values in [0,N-1] are okay for iN

    macro_rules! tests {
        ($single:ident, $eq:ident, $max_rhs:expr, $expect:expr) => { {
            let x = $single(1) << id($single(0));
            assert!($eq(x, $single(1)));
            let x = $single(1) << id($single($max_rhs));
            assert!($eq(x, $single($expect)));
            // high-order bits on LHS are silently discarded without panic.
            let x = $single(3) << id($single($max_rhs));
            assert!($eq(x, $single($expect)));
        } }
    }

    let x = single_i8x16(1) << id(single_i8x16(0));
    assert!(eq_i8x16(x, single_i8x16(1)));
    let x = single_u8x16(1) << id(single_u8x16(0));
    assert!(eq_u8x16(x, single_u8x16(1)));
    let x = single_i8x16(1) << id(single_i8x16(7));
    assert!(eq_i8x16(x, single_i8x16(std::i8::MIN)));
    let x = single_u8x16(1) << id(single_u8x16(7));
    assert!(eq_u8x16(x, single_u8x16(0x80)));
    // high-order bits on LHS are silently discarded without panic.
    let x = single_i8x16(3) << id(single_i8x16(7));
    assert!(eq_i8x16(x, single_i8x16(std::i8::MIN)));
    let x = single_u8x16(3) << id(single_u8x16(7));
    assert!(eq_u8x16(x, single_u8x16(0x80)));

    // above is (approximately) expanded from:
    tests!(single_i8x16, eq_i8x16, 7, std::i8::MIN);
    tests!(single_u8x16, eq_u8x16, 7, 0x80_u8);

    tests!(single_i16x8, eq_i16x8, 15, std::i16::MIN);
    tests!(single_u16x8, eq_u16x8, 15, 0x8000_u16);

    tests!(single_i32x4, eq_i32x4, 31, std::i32::MIN);
    tests!(single_u32x4, eq_u32x4, 31, 0x8000_0000_u32);

    tests!(single_i64x2, eq_i64x2, 63, std::i64::MIN);
    tests!(single_u64x2, eq_u64x2, 63, 0x8000_0000_0000_0000_u64);
}

fn test_right_shift() {
    // negative rhs can panic, but values in [0,N-1] are okay for iN

    macro_rules! tests {
        ($single_i:ident, $eq_i:ident, $single_u:ident, $eq_u:ident,
         $max_rhs:expr, $signbit_i:expr, $highbit_i:expr, $highbit_u:expr) => { {
            let x = $single_i(1) >> id($single_i(0));
            assert!($eq_i(x, $single_i(1)));
            let x = $single_u(1) >> id($single_u(0));
            assert!($eq_u(x, $single_u(1)));
            let x = $single_u($highbit_i) >> id($single_u($max_rhs-1));
            assert!($eq_u(x, $single_u(1)));
            let x = $single_u($highbit_u) >> id($single_u($max_rhs));
            assert!($eq_u(x, $single_u(1)));
            // sign-bit is carried by arithmetic right shift
            let x = $single_i($signbit_i) >> id($single_i($max_rhs));
            assert!($eq_i(x, $single_i(-1)));
            // low-order bits on LHS are silently discarded without panic.
            let x = $single_u($highbit_i + 1) >> id($single_u($max_rhs-1));
            assert!($eq_u(x, $single_u(1)));
            let x = $single_u($highbit_u + 1) >> id($single_u($max_rhs));
            assert!($eq_u(x, $single_u(1)));
            let x = $single_i($signbit_i + 1) >> id($single_i($max_rhs));
            assert!($eq_i(x, $single_i(-1)));
        } }
    }

    tests!(single_i8x16, eq_i8x16, single_u8x16, eq_u8x16,
           7, std::i8::MIN, 0x40_u8, 0x80_u8);
    tests!(single_i16x8, eq_i16x8, single_u16x8, eq_u16x8,
           15, std::i16::MIN, 0x4000_u16, 0x8000_u16);
    tests!(single_i32x4, eq_i32x4, single_u32x4, eq_u32x4,
           31, std::i32::MIN, 0x4000_0000_u32, 0x8000_0000_u32);
    tests!(single_i64x2, eq_i64x2, single_u64x2, eq_u64x2,
           63, std::i64::MIN, 0x4000_0000_0000_0000_u64, 0x8000_0000_0000_0000_u64);
}
