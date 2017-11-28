// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z lower_128bit_ops -C debug_assertions=yes

#![feature(i128_type)]
#![feature(lang_items)]

#[lang="i128_div"]
fn i128_div(_x: i128, _y: i128) -> i128 { 3 }
#[lang="u128_div"]
fn u128_div(_x: u128, _y: u128) -> u128 { 4 }
#[lang="i128_rem"]
fn i128_rem(_x: i128, _y: i128) -> i128 { 5 }
#[lang="u128_rem"]
fn u128_rem(_x: u128, _y: u128) -> u128 { 6 }

#[lang="i128_addo"]
fn i128_addo(_x: i128, _y: i128) -> (i128, bool) { (0, false) }
#[lang="u128_addo"]
fn u128_addo(_x: u128, _y: u128) -> (u128, bool) { (1, false) }
#[lang="i128_subo"]
fn i128_subo(_x: i128, _y: i128) -> (i128, bool) { (2, false) }
#[lang="u128_subo"]
fn u128_subo(_x: u128, _y: u128) -> (u128, bool) { (3, false) }
#[lang="i128_mulo"]
fn i128_mulo(_x: i128, _y: i128) -> (i128, bool) { (4, false) }
#[lang="u128_mulo"]
fn u128_mulo(_x: u128, _y: u128) -> (u128, bool) { (5, false) }
#[lang="i128_shlo"]
fn i128_shlo(_x: i128, _y: u128) -> (i128, bool) { (6, false) }
#[lang="u128_shlo"]
fn u128_shlo(_x: u128, _y: u128) -> (u128, bool) { (6, false) }
#[lang="i128_shro"]
fn i128_shro(_x: i128, _y: u128) -> (i128, bool) { (7, false) }
#[lang="u128_shro"]
fn u128_shro(_x: u128, _y: u128) -> (u128, bool) { (8, false) }

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
//     _2 = const i128_addo(_1, const 1i128) -> bb10;
//     ...
//     _1 = move (_2.0: i128);
//     _3 = const i128_subo(_1, const 2i128) -> bb11;
//     ...
//     _1 = move (_3.0: i128);
//     _4 = const i128_mulo(_1, const 3i128) -> bb12;
//     ...
//     _1 = move (_4.0: i128);
//     ...
//     _1 = const i128_div(_1, const 4i128) -> bb13;
//     ...
//     _1 = const i128_rem(_1, const 5i128) -> bb15;
//     ...
//     _1 = move (_13.0: i128);
//     ...
//     _17 = const 7i32 as u128 (Misc);
//     _14 = const i128_shro(_1, move _17) -> bb16;
//     ...
//     _1 = move (_14.0: i128);
//     ...
//     assert(!move (_2.1: bool), "attempt to add with overflow") -> bb1;
//     ...
//     assert(!move (_3.1: bool), "attempt to subtract with overflow") -> bb2;
//     ...
//     assert(!move (_4.1: bool), "attempt to multiply with overflow") -> bb3;
//     ...
//     assert(!move (_13.1: bool), "attempt to shift left with overflow") -> bb8;
//     ...
//     _16 = const 6i32 as u128 (Misc);
//     _13 = const i128_shlo(_1, move _16) -> bb14;
//     ...
//     assert(!move (_14.1: bool), "attempt to shift right with overflow") -> bb9;
// END rustc.test_signed.Lower128Bit.after.mir

// START rustc.test_unsigned.Lower128Bit.after.mir
//     _2 = const u128_addo(_1, const 1u128) -> bb8;
//     ...
//     _1 = move (_2.0: u128);
//     _3 = const u128_subo(_1, const 2u128) -> bb9;
//     ...
//     _1 = move (_3.0: u128);
//     _4 = const u128_mulo(_1, const 3u128) -> bb10;
//     ...
//     _1 = move (_4.0: u128);
//     ...
//     _1 = const u128_div(_1, const 4u128) -> bb11;
//     ...
//     _1 = const u128_rem(_1, const 5u128) -> bb13;
//     ...
//     _1 = move (_7.0: u128);
//     ...
//     _11 = const 7i32 as u128 (Misc);
//     _8 = const u128_shro(_1, move _11) -> bb14;
//     ...
//     _1 = move (_8.0: u128);
//     ...
//     assert(!move (_2.1: bool), "attempt to add with overflow") -> bb1;
//     ...
//     assert(!move (_3.1: bool), "attempt to subtract with overflow") -> bb2;
//     ...
//     assert(!move (_4.1: bool), "attempt to multiply with overflow") -> bb3;
//     ...
//     assert(!move (_7.1: bool), "attempt to shift left with overflow") -> bb6;
//     ...
//     _10 = const 6i32 as u128 (Misc);
//     _7 = const u128_shlo(_1, move _10) -> bb12;
//     ...
//     assert(!move (_8.1: bool), "attempt to shift right with overflow") -> bb7;
// END rustc.test_unsigned.Lower128Bit.after.mir
