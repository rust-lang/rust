// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z lower_128bit_ops -C debug_assertions=no

#![feature(i128_type)]
#![feature(lang_items)]

#[lang="i128_add"]
fn i128_add(_x: i128, _y: i128) -> i128 { 0 }
#[lang="u128_add"]
fn u128_add(_x: u128, _y: u128) -> u128 { 0 }
#[lang="i128_sub"]
fn i128_sub(_x: i128, _y: i128) -> i128 { 1 }
#[lang="u128_sub"]
fn u128_sub(_x: u128, _y: u128) -> u128 { 1 }
#[lang="i128_mul"]
fn i128_mul(_x: i128, _y: i128) -> i128 { 2 }
#[lang="u128_mul"]
fn u128_mul(_x: u128, _y: u128) -> u128 { 2 }
#[lang="i128_div"]
fn i128_div(_x: i128, _y: i128) -> i128 { 3 }
#[lang="u128_div"]
fn u128_div(_x: u128, _y: u128) -> u128 { 4 }
#[lang="i128_rem"]
fn i128_rem(_x: i128, _y: i128) -> i128 { 5 }
#[lang="u128_rem"]
fn u128_rem(_x: u128, _y: u128) -> u128 { 6 }
#[lang="i128_shl"]
fn i128_shl(_x: i128, _y: u32) -> i128 { 7 }
#[lang="u128_shl"]
fn u128_shl(_x: u128, _y: u32) -> u128 { 7 }
#[lang="i128_shr"]
fn i128_shr(_x: i128, _y: u32) -> i128 { 8 }
#[lang="u128_shr"]
fn u128_shr(_x: u128, _y: u32) -> u128 { 9 }

fn test_signed(mut x: i128) -> i128 {
    x += 1;
    x -= 2;
    x *= 3;
    x /= 4;
    x %= 5;
    x <<= 6;
    x >>= 7;
    x
}

fn test_unsigned(mut x: u128) -> u128 {
    x += 1;
    x -= 2;
    x *= 3;
    x /= 4;
    x %= 5;
    x <<= 6;
    x >>= 7;
    x
}

fn main() {
    test_signed(-200);
    test_unsigned(200);
}

// END RUST SOURCE

// START rustc.test_signed.Lower128Bit.after.mir
//     _1 = const i128_add(_1, const 1i128) -> bb7;
//     ...
//     _1 = const i128_div(_1, const 4i128) -> bb8;
//     ...
//     _1 = const i128_rem(_1, const 5i128) -> bb11;
//     ...
//     _1 = const i128_mul(_1, const 3i128) -> bb5;
//     ...
//     _1 = const i128_sub(_1, const 2i128) -> bb6;
//     ...
//     _11 = const 7i32 as u32 (Misc);
//     _1 = const i128_shr(_1, _11) -> bb9;
//     ...
//     _12 = const 6i32 as u32 (Misc);
//     _1 = const i128_shl(_1, _12) -> bb10;
// END rustc.test_signed.Lower128Bit.after.mir

// START rustc.test_unsigned.Lower128Bit.after.mir
//     _1 = const u128_add(_1, const 1u128) -> bb5;
//     ...
//     _1 = const u128_div(_1, const 4u128) -> bb6;
//     ...
//     _1 = const u128_rem(_1, const 5u128) -> bb9;
//     ...
//     _1 = const u128_mul(_1, const 3u128) -> bb3;
//     ...
//     _1 = const u128_sub(_1, const 2u128) -> bb4;
//     ...
//     _5 = const 7i32 as u32 (Misc);
//     _1 = const u128_shr(_1, _5) -> bb7;
//     ...
//     _6 = const 6i32 as u32 (Misc);
//     _1 = const u128_shl(_1, _6) -> bb8;
// END rustc.test_unsigned.Lower128Bit.after.mir
