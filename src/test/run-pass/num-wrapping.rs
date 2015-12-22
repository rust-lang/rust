// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Test std::num::Wrapping<T> for {uN, iN, usize, isize}
#![feature(op_assign_traits)]

use std::num::Wrapping;
use std::ops::{ AddAssign, SubAssign, MulAssign, DivAssign };
use std::{ i8, i16, i32, i64, isize, u8, u16, u32, u64, usize };

fn main() {
    test_ops();
    test_op_assigns();
}

fn test_ops() {
    // TODO(ubsan): test impl Op<T> for Wrapping<T>, when/if it is added
    // TODO(ubsan): check for wrapping Rem after impl-ing Rem
    assert_eq!(Wrapping(i8::MAX) + Wrapping(1), Wrapping(i8::MIN));
    assert_eq!(Wrapping(i16::MAX) + Wrapping(1), Wrapping(i16::MIN));
    assert_eq!(Wrapping(i32::MAX) + Wrapping(1), Wrapping(i32::MIN));
    assert_eq!(Wrapping(i64::MAX) + Wrapping(1), Wrapping(i64::MIN));
    assert_eq!(Wrapping(isize::MAX) + Wrapping(1), Wrapping(isize::MIN));

    assert_eq!(Wrapping(u8::MAX) + Wrapping(1), Wrapping(0));
    assert_eq!(Wrapping(u16::MAX) + Wrapping(1), Wrapping(0));
    assert_eq!(Wrapping(u32::MAX) + Wrapping(1), Wrapping(0));
    assert_eq!(Wrapping(u64::MAX) + Wrapping(1), Wrapping(0));
    assert_eq!(Wrapping(usize::MAX) + Wrapping(1), Wrapping(0));


    assert_eq!(Wrapping(i8::MIN) - Wrapping(1), Wrapping(i8::MAX));
    assert_eq!(Wrapping(i16::MIN) - Wrapping(1), Wrapping(i16::MAX));
    assert_eq!(Wrapping(i32::MIN) - Wrapping(1), Wrapping(i32::MAX));
    assert_eq!(Wrapping(i64::MIN) - Wrapping(1), Wrapping(i64::MAX));
    assert_eq!(Wrapping(isize::MIN) - Wrapping(1), Wrapping(isize::MAX));

    assert_eq!(Wrapping(0u8) - Wrapping(1), Wrapping(u8::MAX));
    assert_eq!(Wrapping(0u16) - Wrapping(1), Wrapping(u16::MAX));
    assert_eq!(Wrapping(0u32) - Wrapping(1), Wrapping(u32::MAX));
    assert_eq!(Wrapping(0u64) - Wrapping(1), Wrapping(u64::MAX));
    assert_eq!(Wrapping(0usize) - Wrapping(1), Wrapping(usize::MAX));


    assert_eq!(Wrapping(i8::MAX) * Wrapping(2), Wrapping(-2));
    assert_eq!(Wrapping(i8::MAX) * Wrapping(2), Wrapping(-2));
    assert_eq!(Wrapping(i8::MAX) * Wrapping(2), Wrapping(-2));
    assert_eq!(Wrapping(i8::MAX) * Wrapping(2), Wrapping(-2));
    assert_eq!(Wrapping(i8::MAX) * Wrapping(2), Wrapping(-2));

    assert_eq!(Wrapping(u8::MAX) * Wrapping(2), Wrapping(u8::MAX - 1));
    assert_eq!(Wrapping(u16::MAX) * Wrapping(2), Wrapping(u16::MAX - 1));
    assert_eq!(Wrapping(u32::MAX) * Wrapping(2), Wrapping(u32::MAX - 1));
    assert_eq!(Wrapping(u64::MAX) * Wrapping(2), Wrapping(u64::MAX - 1));
    assert_eq!(Wrapping(usize::MAX) * Wrapping(2), Wrapping(usize::MAX - 1));


    assert_eq!(Wrapping(i8::MIN) / Wrapping(-1), Wrapping(i8::MIN));
    assert_eq!(Wrapping(i16::MIN) / Wrapping(-1), Wrapping(i16::MIN));
    assert_eq!(Wrapping(i32::MIN) / Wrapping(-1), Wrapping(i32::MIN));
    assert_eq!(Wrapping(i64::MIN) / Wrapping(-1), Wrapping(i64::MIN));
    assert_eq!(Wrapping(isize::MIN) / Wrapping(-1), Wrapping(isize::MIN));
}

fn test_op_assigns() {
    // TODO(ubsan): test impl OpAssign<T> for Wrapping<T>, when/if it is added
    // TODO(ubsan): test RemAssign after impl-ing Rem
    macro_rules! op_assign_test {
        ($op:ident ($initial:expr, $rhs:expr) == $ans:expr) => {{
            let mut tmp = $initial;
            tmp.$op($rhs);
            assert_eq!(tmp, $ans);
        }}
    }
    op_assign_test!(add_assign(Wrapping(i8::MAX), (Wrapping(1))) == Wrapping(i8::MIN));
    op_assign_test!(add_assign(Wrapping(i16::MAX), (Wrapping(1))) == Wrapping(i16::MIN));
    op_assign_test!(add_assign(Wrapping(i32::MAX), (Wrapping(1))) == Wrapping(i32::MIN));
    op_assign_test!(add_assign(Wrapping(i64::MAX), (Wrapping(1))) == Wrapping(i64::MIN));
    op_assign_test!(add_assign(Wrapping(isize::MAX), (Wrapping(1))) == Wrapping(isize::MIN));

    op_assign_test!(add_assign(Wrapping(u8::MAX), (Wrapping(1))) == Wrapping(u8::MIN));
    op_assign_test!(add_assign(Wrapping(u16::MAX), (Wrapping(1))) == Wrapping(u16::MIN));
    op_assign_test!(add_assign(Wrapping(u32::MAX), (Wrapping(1))) == Wrapping(u32::MIN));
    op_assign_test!(add_assign(Wrapping(u64::MAX), (Wrapping(1))) == Wrapping(u64::MIN));
    op_assign_test!(add_assign(Wrapping(usize::MAX), (Wrapping(1))) == Wrapping(usize::MIN));


    op_assign_test!(sub_assign(Wrapping(i8::MIN), (Wrapping(1))) == Wrapping(i8::MAX));
    op_assign_test!(sub_assign(Wrapping(i16::MIN), (Wrapping(1))) == Wrapping(i16::MAX));
    op_assign_test!(sub_assign(Wrapping(i32::MIN), (Wrapping(1))) == Wrapping(i32::MAX));
    op_assign_test!(sub_assign(Wrapping(i64::MIN), (Wrapping(1))) == Wrapping(i64::MAX));
    op_assign_test!(sub_assign(Wrapping(isize::MIN), (Wrapping(1))) == Wrapping(isize::MAX));

    op_assign_test!(sub_assign(Wrapping(u8::MIN), (Wrapping(1))) == Wrapping(u8::MAX));
    op_assign_test!(sub_assign(Wrapping(u16::MIN), (Wrapping(1))) == Wrapping(u16::MAX));
    op_assign_test!(sub_assign(Wrapping(u32::MIN), (Wrapping(1))) == Wrapping(u32::MAX));
    op_assign_test!(sub_assign(Wrapping(u64::MIN), (Wrapping(1))) == Wrapping(u64::MAX));
    op_assign_test!(sub_assign(Wrapping(usize::MIN), (Wrapping(1))) == Wrapping(usize::MAX));


    op_assign_test!(mul_assign(Wrapping(i8::MIN), (Wrapping(1))) == Wrapping(i8::MAX));
    op_assign_test!(mul_assign(Wrapping(i16::MIN), (Wrapping(1))) == Wrapping(i16::MAX));
    op_assign_test!(mul_assign(Wrapping(i32::MIN), (Wrapping(1))) == Wrapping(i32::MAX));
    op_assign_test!(mul_assign(Wrapping(i64::MIN), (Wrapping(1))) == Wrapping(i64::MAX));
    op_assign_test!(mul_assign(Wrapping(isize::MIN), (Wrapping(1))) == Wrapping(isize::MAX));

    op_assign_test!(mul_assign(Wrapping(0u8), (Wrapping(1))) == Wrapping(u8::MAX));
    op_assign_test!(mul_assign(Wrapping(0u16), (Wrapping(1))) == Wrapping(u16::MAX));
    op_assign_test!(mul_assign(Wrapping(0u32), (Wrapping(1))) == Wrapping(u32::MAX));
    op_assign_test!(mul_assign(Wrapping(0u64), (Wrapping(1))) == Wrapping(u64::MAX));
    op_assign_test!(mul_assign(Wrapping(0usize), (Wrapping(1))) == Wrapping(usize::MAX));


    op_assign_test!(div_assign(Wrapping(i8::MIN), (Wrapping(-1))) == Wrapping(i8::MIN));
    op_assign_test!(div_assign(Wrapping(i16::MIN), (Wrapping(-1))) == Wrapping(i16::MIN));
    op_assign_test!(div_assign(Wrapping(i32::MIN), (Wrapping(-1))) == Wrapping(i32::MIN));
    op_assign_test!(div_assign(Wrapping(i64::MIN), (Wrapping(-1))) == Wrapping(i64::MIN));
    op_assign_test!(div_assign(Wrapping(isize::MIN), (Wrapping(-1))) == Wrapping(isize::MIN));
}
