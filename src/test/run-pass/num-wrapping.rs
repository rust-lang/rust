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
// compile-flags: -C debug-assertions
//
// Test std::num::Wrapping<T> for {uN, iN, usize, isize}

#![feature(op_assign_traits, num_bits_bytes)]

use std::num::Wrapping;
use std::ops::{
    Add, Sub, Mul, Div, Rem, BitXor, BitOr, BitAnd,
    AddAssign, SubAssign, MulAssign, DivAssign, RemAssign, BitXorAssign, BitOrAssign, BitAndAssign,
    Shl, Shr, ShlAssign, ShrAssign
};
use std::{i8, i16, i32, i64, isize, u8, u16, u32, u64, usize};

use std::test::black_box;

fn main() {
    test_ops();
    test_op_assigns();
    test_sh_ops();
    test_sh_op_assigns();
}

fn test_ops() {
    macro_rules! op_test {
        ($op:ident ($lhs:expr, $rhs:expr) == $ans:expr) => {
            assert_eq!(black_box(Wrapping($lhs).$op(Wrapping($rhs))), Wrapping($ans));
            assert_eq!(black_box(Wrapping($lhs).$op($rhs)), Wrapping($ans));
        }
    }

    op_test!(add(i8::MAX, 1) == i8::MIN);
    op_test!(add(i16::MAX, 1) == i16::MIN);
    op_test!(add(i32::MAX, 1) == i32::MIN);
    op_test!(add(i64::MAX, 1) == i64::MIN);
    op_test!(add(isize::MAX, 1) == isize::MIN);

    op_test!(add(u8::MAX, 1) == 0);
    op_test!(add(u16::MAX, 1) == 0);
    op_test!(add(u32::MAX, 1) == 0);
    op_test!(add(u64::MAX, 1) == 0);
    op_test!(add(usize::MAX, 1) == 0);


    op_test!(sub(i8::MIN, 1) == i8::MAX);
    op_test!(sub(i16::MIN, 1) == i16::MAX);
    op_test!(sub(i32::MIN, 1) == i32::MAX);
    op_test!(sub(i64::MIN, 1) == i64::MAX);
    op_test!(sub(isize::MIN, 1) == isize::MAX);

    op_test!(sub(0u8, 1) == u8::MAX);
    op_test!(sub(0u16, 1) == u16::MAX);
    op_test!(sub(0u32, 1) == u32::MAX);
    op_test!(sub(0u64, 1) == u64::MAX);
    op_test!(sub(0usize, 1) == usize::MAX);


    op_test!(mul(i8::MAX, 2) == -2);
    op_test!(mul(i16::MAX, 2) == -2);
    op_test!(mul(i32::MAX, 2) == -2);
    op_test!(mul(i64::MAX, 2) == -2);
    op_test!(mul(isize::MAX, 2) == -2);

    op_test!(mul(u8::MAX, 2) == u8::MAX - 1);
    op_test!(mul(u16::MAX, 2) == u16::MAX - 1);
    op_test!(mul(u32::MAX, 2) == u32::MAX - 1);
    op_test!(mul(u64::MAX, 2) == u64::MAX - 1);
    op_test!(mul(usize::MAX, 2) == usize::MAX - 1);


    op_test!(div(i8::MIN, -1) == i8::MIN);
    op_test!(div(i16::MIN, -1) == i16::MIN);
    op_test!(div(i32::MIN, -1) == i32::MIN);
    op_test!(div(i64::MIN, -1) == i64::MIN);
    op_test!(div(isize::MIN, -1) == isize::MIN);


    op_test!(rem(i8::MIN, -1) == 0);
    op_test!(rem(i16::MIN, -1) == 0);
    op_test!(rem(i32::MIN, -1) == 0);
    op_test!(rem(i64::MIN, -1) == 0);
    op_test!(rem(isize::MIN, -1) == 0);

    // these are not that interesting, just testing to make sure they are implemented correctly
    op_test!(bitxor(0b101010i8, 0b100110) == 0b001100);
    op_test!(bitxor(0b101010i16, 0b100110) == 0b001100);
    op_test!(bitxor(0b101010i32, 0b100110) == 0b001100);
    op_test!(bitxor(0b101010i64, 0b100110) == 0b001100);
    op_test!(bitxor(0b101010isize, 0b100110) == 0b001100);

    op_test!(bitxor(0b101010u8, 0b100110) == 0b001100);
    op_test!(bitxor(0b101010u16, 0b100110) == 0b001100);
    op_test!(bitxor(0b101010u32, 0b100110) == 0b001100);
    op_test!(bitxor(0b101010u64, 0b100110) == 0b001100);
    op_test!(bitxor(0b101010usize, 0b100110) == 0b001100);


    op_test!(bitor(0b101010i8, 0b100110) == 0b101110);
    op_test!(bitor(0b101010i16, 0b100110) == 0b101110);
    op_test!(bitor(0b101010i32, 0b100110) == 0b101110);
    op_test!(bitor(0b101010i64, 0b100110) == 0b101110);
    op_test!(bitor(0b101010isize, 0b100110) == 0b101110);

    op_test!(bitor(0b101010u8, 0b100110) == 0b101110);
    op_test!(bitor(0b101010u16, 0b100110) == 0b101110);
    op_test!(bitor(0b101010u32, 0b100110) == 0b101110);
    op_test!(bitor(0b101010u64, 0b100110) == 0b101110);
    op_test!(bitor(0b101010usize, 0b100110) == 0b101110);


    op_test!(bitand(0b101010i8, 0b100110) == 0b100010);
    op_test!(bitand(0b101010i16, 0b100110) == 0b100010);
    op_test!(bitand(0b101010i32, 0b100110) == 0b100010);
    op_test!(bitand(0b101010i64, 0b100110) == 0b100010);
    op_test!(bitand(0b101010isize, 0b100110) == 0b100010);

    op_test!(bitand(0b101010u8, 0b100110) == 0b100010);
    op_test!(bitand(0b101010u16, 0b100110) == 0b100010);
    op_test!(bitand(0b101010u32, 0b100110) == 0b100010);
    op_test!(bitand(0b101010u64, 0b100110) == 0b100010);
    op_test!(bitand(0b101010usize, 0b100110) == 0b100010);
}

fn test_op_assigns() {
    macro_rules! op_assign_test {
        ($op:ident ($initial:expr, $rhs:expr) == $ans:expr) => {
            {
                let mut tmp = Wrapping($initial);
                tmp = black_box(tmp);
                tmp.$op(Wrapping($rhs));
                assert_eq!(black_box(tmp), Wrapping($ans));
            }
            {
                let mut tmp = Wrapping($initial);
                tmp = black_box(tmp);
                tmp.$op($rhs);
                assert_eq!(black_box(tmp), Wrapping($ans));
            }
        }
    }
    op_assign_test!(add_assign(i8::MAX, 1) == i8::MIN);
    op_assign_test!(add_assign(i16::MAX, 1) == i16::MIN);
    op_assign_test!(add_assign(i32::MAX, 1) == i32::MIN);
    op_assign_test!(add_assign(i64::MAX, 1) == i64::MIN);
    op_assign_test!(add_assign(isize::MAX, 1) == isize::MIN);

    op_assign_test!(add_assign(u8::MAX, 1) == u8::MIN);
    op_assign_test!(add_assign(u16::MAX, 1) == u16::MIN);
    op_assign_test!(add_assign(u32::MAX, 1) == u32::MIN);
    op_assign_test!(add_assign(u64::MAX, 1) == u64::MIN);
    op_assign_test!(add_assign(usize::MAX, 1) == usize::MIN);


    op_assign_test!(sub_assign(i8::MIN, 1) == i8::MAX);
    op_assign_test!(sub_assign(i16::MIN, 1) == i16::MAX);
    op_assign_test!(sub_assign(i32::MIN, 1) == i32::MAX);
    op_assign_test!(sub_assign(i64::MIN, 1) == i64::MAX);
    op_assign_test!(sub_assign(isize::MIN, 1) == isize::MAX);

    op_assign_test!(sub_assign(u8::MIN, 1) == u8::MAX);
    op_assign_test!(sub_assign(u16::MIN, 1) == u16::MAX);
    op_assign_test!(sub_assign(u32::MIN, 1) == u32::MAX);
    op_assign_test!(sub_assign(u64::MIN, 1) == u64::MAX);
    op_assign_test!(sub_assign(usize::MIN, 1) == usize::MAX);


    op_assign_test!(mul_assign(i8::MAX, 2) == -2);
    op_assign_test!(mul_assign(i16::MAX, 2) == -2);
    op_assign_test!(mul_assign(i32::MAX, 2) == -2);
    op_assign_test!(mul_assign(i64::MAX, 2) == -2);
    op_assign_test!(mul_assign(isize::MAX, 2) == -2);

    op_assign_test!(mul_assign(u8::MAX, 2) == u8::MAX - 1);
    op_assign_test!(mul_assign(u16::MAX, 2) == u16::MAX - 1);
    op_assign_test!(mul_assign(u32::MAX, 2) == u32::MAX - 1);
    op_assign_test!(mul_assign(u64::MAX, 2) == u64::MAX - 1);
    op_assign_test!(mul_assign(usize::MAX, 2) == usize::MAX - 1);


    op_assign_test!(div_assign(i8::MIN, -1) == i8::MIN);
    op_assign_test!(div_assign(i16::MIN, -1) == i16::MIN);
    op_assign_test!(div_assign(i32::MIN, -1) == i32::MIN);
    op_assign_test!(div_assign(i64::MIN, -1) == i64::MIN);
    op_assign_test!(div_assign(isize::MIN, -1) == isize::MIN);


    op_assign_test!(rem_assign(i8::MIN, -1) == 0);
    op_assign_test!(rem_assign(i16::MIN, -1) == 0);
    op_assign_test!(rem_assign(i32::MIN, -1) == 0);
    op_assign_test!(rem_assign(i64::MIN, -1) == 0);
    op_assign_test!(rem_assign(isize::MIN, -1) == 0);


    // these are not that interesting, just testing to make sure they are implemented correctly
    op_assign_test!(bitxor_assign(0b101010i8, 0b100110) == 0b001100);
    op_assign_test!(bitxor_assign(0b101010i16, 0b100110) == 0b001100);
    op_assign_test!(bitxor_assign(0b101010i32, 0b100110) == 0b001100);
    op_assign_test!(bitxor_assign(0b101010i64, 0b100110) == 0b001100);
    op_assign_test!(bitxor_assign(0b101010isize, 0b100110) == 0b001100);

    op_assign_test!(bitxor_assign(0b101010u8, 0b100110) == 0b001100);
    op_assign_test!(bitxor_assign(0b101010u16, 0b100110) == 0b001100);
    op_assign_test!(bitxor_assign(0b101010u32, 0b100110) == 0b001100);
    op_assign_test!(bitxor_assign(0b101010u64, 0b100110) == 0b001100);
    op_assign_test!(bitxor_assign(0b101010usize, 0b100110) == 0b001100);


    op_assign_test!(bitor_assign(0b101010i8, 0b100110) == 0b101110);
    op_assign_test!(bitor_assign(0b101010i16, 0b100110) == 0b101110);
    op_assign_test!(bitor_assign(0b101010i32, 0b100110) == 0b101110);
    op_assign_test!(bitor_assign(0b101010i64, 0b100110) == 0b101110);
    op_assign_test!(bitor_assign(0b101010isize, 0b100110) == 0b101110);

    op_assign_test!(bitor_assign(0b101010u8, 0b100110) == 0b101110);
    op_assign_test!(bitor_assign(0b101010u16, 0b100110) == 0b101110);
    op_assign_test!(bitor_assign(0b101010u32, 0b100110) == 0b101110);
    op_assign_test!(bitor_assign(0b101010u64, 0b100110) == 0b101110);
    op_assign_test!(bitor_assign(0b101010usize, 0b100110) == 0b101110);


    op_assign_test!(bitand_assign(0b101010i8, 0b100110) == 0b100010);
    op_assign_test!(bitand_assign(0b101010i16, 0b100110) == 0b100010);
    op_assign_test!(bitand_assign(0b101010i32, 0b100110) == 0b100010);
    op_assign_test!(bitand_assign(0b101010i64, 0b100110) == 0b100010);
    op_assign_test!(bitand_assign(0b101010isize, 0b100110) == 0b100010);

    op_assign_test!(bitand_assign(0b101010u8, 0b100110) == 0b100010);
    op_assign_test!(bitand_assign(0b101010u16, 0b100110) == 0b100010);
    op_assign_test!(bitand_assign(0b101010u32, 0b100110) == 0b100010);
    op_assign_test!(bitand_assign(0b101010u64, 0b100110) == 0b100010);
    op_assign_test!(bitand_assign(0b101010usize, 0b100110) == 0b100010);
}

fn test_sh_ops() {
    macro_rules! sh_test {
        ($op:ident ($lhs:expr, $rhs:expr) == $ans:expr) => {
            assert_eq!(black_box(Wrapping($lhs).$op($rhs)), Wrapping($ans));
        }
    }
    sh_test!(shl(i8::MAX, (i8::BITS + 1) as u8) == -2);
    sh_test!(shl(i16::MAX, (i16::BITS + 1) as u8) == -2);
    sh_test!(shl(i32::MAX, (i32::BITS + 1) as u8) == -2);
    sh_test!(shl(i64::MAX, (i64::BITS + 1) as u8) == -2);
    sh_test!(shl(isize::MAX, (isize::BITS + 1) as u8) == -2);

    sh_test!(shl(u8::MAX, (u8::BITS + 1) as i8) == u8::MAX - 1);
    sh_test!(shl(u16::MAX, (u16::BITS + 1) as i8) == u16::MAX - 1);
    sh_test!(shl(u32::MAX, (u32::BITS + 1) as i8) == u32::MAX - 1);
    sh_test!(shl(u64::MAX, (u64::BITS + 1) as i8) == u64::MAX - 1);
    sh_test!(shl(usize::MAX, (usize::BITS + 1) as i8) == usize::MAX - 1);

    sh_test!(shl(i8::MAX, (i8::BITS + 1) as i16) == -2);
    sh_test!(shl(i16::MAX, (i16::BITS + 1) as i16) == -2);
    sh_test!(shl(i32::MAX, (i32::BITS + 1) as i16) == -2);
    sh_test!(shl(i64::MAX, (i64::BITS + 1) as i16) == -2);
    sh_test!(shl(isize::MAX, (isize::BITS + 1) as i16) == -2);

    sh_test!(shl(u8::MAX, (u8::BITS + 1) as i16) == u8::MAX - 1);
    sh_test!(shl(u16::MAX, (u16::BITS + 1) as i16) == u16::MAX - 1);
    sh_test!(shl(u32::MAX, (u32::BITS + 1) as i16) == u32::MAX - 1);
    sh_test!(shl(u64::MAX, (u64::BITS + 1) as i16) == u64::MAX - 1);
    sh_test!(shl(usize::MAX, (usize::BITS + 1) as i16) == usize::MAX - 1);

    sh_test!(shl(i8::MAX, (i8::BITS + 1) as u16) == -2);
    sh_test!(shl(i16::MAX, (i16::BITS + 1) as u16) == -2);
    sh_test!(shl(i32::MAX, (i32::BITS + 1) as u16) == -2);
    sh_test!(shl(i64::MAX, (i64::BITS + 1) as u16) == -2);
    sh_test!(shl(isize::MAX, (isize::BITS + 1) as u16) == -2);

    sh_test!(shl(u8::MAX, (u8::BITS + 1) as u16) == u8::MAX - 1);
    sh_test!(shl(u16::MAX, (u16::BITS + 1) as u16) == u16::MAX - 1);
    sh_test!(shl(u32::MAX, (u32::BITS + 1) as u16) == u32::MAX - 1);
    sh_test!(shl(u64::MAX, (u64::BITS + 1) as u16) == u64::MAX - 1);
    sh_test!(shl(usize::MAX, (usize::BITS + 1) as u16) == usize::MAX - 1);

    sh_test!(shl(i8::MAX, (i8::BITS + 1) as i32) == -2);
    sh_test!(shl(i16::MAX, (i16::BITS + 1) as i32) == -2);
    sh_test!(shl(i32::MAX, (i32::BITS + 1) as i32) == -2);
    sh_test!(shl(i64::MAX, (i64::BITS + 1) as i32) == -2);
    sh_test!(shl(isize::MAX, (isize::BITS + 1) as i32) == -2);

    sh_test!(shl(u8::MAX, (u8::BITS + 1) as i32) == u8::MAX - 1);
    sh_test!(shl(u16::MAX, (u16::BITS + 1) as i32) == u16::MAX - 1);
    sh_test!(shl(u32::MAX, (u32::BITS + 1) as i32) == u32::MAX - 1);
    sh_test!(shl(u64::MAX, (u64::BITS + 1) as i32) == u64::MAX - 1);
    sh_test!(shl(usize::MAX, (usize::BITS + 1) as i32) == usize::MAX - 1);

    sh_test!(shl(i8::MAX, (i8::BITS + 1) as u32) == -2);
    sh_test!(shl(i16::MAX, (i16::BITS + 1) as u32) == -2);
    sh_test!(shl(i32::MAX, (i32::BITS + 1) as u32) == -2);
    sh_test!(shl(i64::MAX, (i64::BITS + 1) as u32) == -2);
    sh_test!(shl(isize::MAX, (isize::BITS + 1) as u32) == -2);

    sh_test!(shl(u8::MAX, (u8::BITS + 1) as u32) == u8::MAX - 1);
    sh_test!(shl(u16::MAX, (u16::BITS + 1) as u32) == u16::MAX - 1);
    sh_test!(shl(u32::MAX, (u32::BITS + 1) as u32) == u32::MAX - 1);
    sh_test!(shl(u64::MAX, (u64::BITS + 1) as u32) == u64::MAX - 1);
    sh_test!(shl(usize::MAX, (usize::BITS + 1) as u32) == usize::MAX - 1);

    sh_test!(shl(i8::MAX, (i8::BITS + 1) as i64) == -2);
    sh_test!(shl(i16::MAX, (i16::BITS + 1) as i64) == -2);
    sh_test!(shl(i32::MAX, (i32::BITS + 1) as i64) == -2);
    sh_test!(shl(i64::MAX, (i64::BITS + 1) as i64) == -2);
    sh_test!(shl(isize::MAX, (isize::BITS + 1) as i64) == -2);

    sh_test!(shl(u8::MAX, (u8::BITS + 1) as i64) == u8::MAX - 1);
    sh_test!(shl(u16::MAX, (u16::BITS + 1) as i64) == u16::MAX - 1);
    sh_test!(shl(u32::MAX, (u32::BITS + 1) as i64) == u32::MAX - 1);
    sh_test!(shl(u64::MAX, (u64::BITS + 1) as i64) == u64::MAX - 1);
    sh_test!(shl(usize::MAX, (usize::BITS + 1) as i64) == usize::MAX - 1);

    sh_test!(shl(i8::MAX, (i8::BITS + 1) as u64) == -2);
    sh_test!(shl(i16::MAX, (i16::BITS + 1) as u64) == -2);
    sh_test!(shl(i32::MAX, (i32::BITS + 1) as u64) == -2);
    sh_test!(shl(i64::MAX, (i64::BITS + 1) as u64) == -2);
    sh_test!(shl(isize::MAX, (isize::BITS + 1) as u64) == -2);

    sh_test!(shl(u8::MAX, (u8::BITS + 1) as u64) == u8::MAX - 1);
    sh_test!(shl(u16::MAX, (u16::BITS + 1) as u64) == u16::MAX - 1);
    sh_test!(shl(u32::MAX, (u32::BITS + 1) as u64) == u32::MAX - 1);
    sh_test!(shl(u64::MAX, (u64::BITS + 1) as u64) == u64::MAX - 1);
    sh_test!(shl(usize::MAX, (usize::BITS + 1) as u64) == usize::MAX - 1);

    sh_test!(shl(i8::MAX, (i8::BITS + 1) as isize) == -2);
    sh_test!(shl(i16::MAX, (i16::BITS + 1) as isize) == -2);
    sh_test!(shl(i32::MAX, (i32::BITS + 1) as isize) == -2);
    sh_test!(shl(i64::MAX, (i64::BITS + 1) as isize) == -2);
    sh_test!(shl(isize::MAX, (isize::BITS + 1) as isize) == -2);

    sh_test!(shl(u8::MAX, (u8::BITS + 1) as isize) == u8::MAX - 1);
    sh_test!(shl(u16::MAX, (u16::BITS + 1) as isize) == u16::MAX - 1);
    sh_test!(shl(u32::MAX, (u32::BITS + 1) as isize) == u32::MAX - 1);
    sh_test!(shl(u64::MAX, (u64::BITS + 1) as isize) == u64::MAX - 1);
    sh_test!(shl(usize::MAX, (usize::BITS + 1) as isize) == usize::MAX - 1);

    sh_test!(shl(i8::MAX, (i8::BITS + 1) as usize) == -2);
    sh_test!(shl(i16::MAX, (i16::BITS + 1) as usize) == -2);
    sh_test!(shl(i32::MAX, (i32::BITS + 1) as usize) == -2);
    sh_test!(shl(i64::MAX, (i64::BITS + 1) as usize) == -2);
    sh_test!(shl(isize::MAX, (isize::BITS + 1) as usize) == -2);

    sh_test!(shl(u8::MAX, (u8::BITS + 1) as usize) == u8::MAX - 1);
    sh_test!(shl(u16::MAX, (u16::BITS + 1) as usize) == u16::MAX - 1);
    sh_test!(shl(u32::MAX, (u32::BITS + 1) as usize) == u32::MAX - 1);
    sh_test!(shl(u64::MAX, (u64::BITS + 1) as usize) == u64::MAX - 1);
    sh_test!(shl(usize::MAX, (usize::BITS + 1) as usize) == usize::MAX - 1);


    sh_test!(shr(i8::MAX, i8::BITS + 1) == i8::MAX / 2);
    sh_test!(shr(i16::MAX, i16::BITS + 1) == i16::MAX / 2);
    sh_test!(shr(i32::MAX, i32::BITS + 1) == i32::MAX / 2);
    sh_test!(shr(i64::MAX, i64::BITS + 1) == i64::MAX / 2);
    sh_test!(shr(isize::MAX, isize::BITS + 1) == isize::MAX / 2);

    sh_test!(shr(u8::MAX, u8::BITS + 1) == u8::MAX / 2);
    sh_test!(shr(u16::MAX, u16::BITS + 1) == u16::MAX / 2);
    sh_test!(shr(u32::MAX, u32::BITS + 1) == u32::MAX / 2);
    sh_test!(shr(u64::MAX, u64::BITS + 1) == u64::MAX / 2);
    sh_test!(shr(usize::MAX, usize::BITS + 1) == usize::MAX / 2);
}

fn test_sh_op_assigns() {
    macro_rules! sh_assign_test {
        ($op:ident ($initial:expr, $rhs:expr) == $ans:expr) => {{
            let mut tmp = Wrapping($initial);
            tmp = black_box(tmp);
            tmp.$op($rhs);
            assert_eq!(black_box(tmp), Wrapping($ans));
        }}
    }
    sh_assign_test!(shl_assign(i8::MAX, i8::BITS + 1) == -2);
    sh_assign_test!(shl_assign(i16::MAX, i16::BITS + 1) == -2);
    sh_assign_test!(shl_assign(i32::MAX, i32::BITS + 1) == -2);
    sh_assign_test!(shl_assign(i64::MAX, i64::BITS + 1) == -2);
    sh_assign_test!(shl_assign(isize::MAX, isize::BITS + 1) == -2);

    sh_assign_test!(shl_assign(u8::MAX, u8::BITS + 1) == u8::MAX - 1);
    sh_assign_test!(shl_assign(u16::MAX, u16::BITS + 1) == u16::MAX - 1);
    sh_assign_test!(shl_assign(u32::MAX, u32::BITS + 1) == u32::MAX - 1);
    sh_assign_test!(shl_assign(u64::MAX, u64::BITS + 1) == u64::MAX - 1);
    sh_assign_test!(shl_assign(usize::MAX, usize::BITS + 1) == usize::MAX - 1);


    sh_assign_test!(shr_assign(i8::MAX, i8::BITS + 1) == i8::MAX / 2);
    sh_assign_test!(shr_assign(i16::MAX, i16::BITS + 1) == i16::MAX / 2);
    sh_assign_test!(shr_assign(i32::MAX, i32::BITS + 1) == i32::MAX / 2);
    sh_assign_test!(shr_assign(i64::MAX, i64::BITS + 1) == i64::MAX / 2);
    sh_assign_test!(shr_assign(isize::MAX, isize::BITS + 1) == isize::MAX / 2);

    sh_assign_test!(shr_assign(u8::MAX, u8::BITS + 1) == u8::MAX / 2);
    sh_assign_test!(shr_assign(u16::MAX, u16::BITS + 1) == u16::MAX / 2);
    sh_assign_test!(shr_assign(u32::MAX, u32::BITS + 1) == u32::MAX / 2);
    sh_assign_test!(shr_assign(u64::MAX, u64::BITS + 1) == u64::MAX / 2);
    sh_assign_test!(shr_assign(usize::MAX, usize::BITS + 1) == usize::MAX / 2);
}
