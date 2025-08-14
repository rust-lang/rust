//@ run-pass
#![allow(unused_macros)]

//@ compile-flags: -C debug-assertions
//
// Test std::num::Wrapping<T> for {uN, iN, usize, isize}

#![feature(test)]

extern crate test;

use std::num::Wrapping;
use std::ops::{
    Add, Sub, Mul, Div, Rem, BitXor, BitOr, BitAnd,
    AddAssign, SubAssign, MulAssign, DivAssign, RemAssign, BitXorAssign, BitOrAssign, BitAndAssign,
    Shl, Shr, ShlAssign, ShrAssign
};
use test::black_box;

macro_rules! int_modules {
    ($(($name:ident, $size:expr),)*) => ($(
        mod $name {
            pub const BITS: usize = $size;
            pub const MAX: $name = $name::MAX;
            pub const MIN: $name = $name::MIN;
        }
    )*)
}

int_modules! {
    (i8, 8),
    (i16, 16),
    (i32, 32),
    (i64, 64),
    (u8, 8),
    (u16, 16),
    (u32, 32),
    (u64, 64),
}

#[cfg(target_pointer_width = "32")]
int_modules! {
    (isize, 32),
    (usize, 32),
}

#[cfg(target_pointer_width = "64")]
int_modules! {
    (isize, 64),
    (usize, 64),
}

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
            // FIXME(30524): uncomment this test when it's implemented
            // assert_eq!(black_box(Wrapping($lhs).$op($rhs)), Wrapping($ans));
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

            // also test that a &Wrapping<T> right-hand side is possible
            {
                let mut tmp = Wrapping($initial);
                tmp = black_box(tmp);
                tmp.$op(&Wrapping($rhs));
                assert_eq!(black_box(tmp), Wrapping($ans));
            }

            // FIXME(30524): uncomment this test
            /*
            {
                let mut tmp = Wrapping($initial);
                tmp = black_box(tmp);
                tmp.$op($rhs);
                assert_eq!(black_box(tmp), Wrapping($ans));
            }
            */
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
    // NOTE: This will break for i8 if we ever get i/u128
    macro_rules! sh_test_all {
        ($t:ty) => {
            sh_test!(shl(i8::MAX, (i8::BITS + 1) as $t) == -2);
            sh_test!(shl(i16::MAX, (i16::BITS + 1) as $t) == -2);
            sh_test!(shl(i32::MAX, (i32::BITS + 1) as $t) == -2);
            sh_test!(shl(i64::MAX, (i64::BITS + 1) as $t) == -2);
            sh_test!(shl(isize::MAX, (isize::BITS + 1) as $t) == -2);

            sh_test!(shl(u8::MAX, (u8::BITS + 1) as $t) == u8::MAX - 1);
            sh_test!(shl(u16::MAX, (u16::BITS + 1) as $t) == u16::MAX - 1);
            sh_test!(shl(u32::MAX, (u32::BITS + 1) as $t) == u32::MAX - 1);
            sh_test!(shl(u64::MAX, (u64::BITS + 1) as $t) == u64::MAX - 1);
            sh_test!(shl(usize::MAX, (usize::BITS + 1) as $t) == usize::MAX - 1);


            sh_test!(shr(i8::MAX, (i8::BITS + 1) as $t) == i8::MAX / 2);
            sh_test!(shr(i16::MAX, (i16::BITS + 1) as $t) == i16::MAX / 2);
            sh_test!(shr(i32::MAX, (i32::BITS + 1) as $t) == i32::MAX / 2);
            sh_test!(shr(i64::MAX, (i64::BITS + 1) as $t) == i64::MAX / 2);
            sh_test!(shr(isize::MAX, (isize::BITS + 1) as $t) == isize::MAX / 2);

            sh_test!(shr(u8::MAX, (u8::BITS + 1) as $t) == u8::MAX / 2);
            sh_test!(shr(u16::MAX, (u16::BITS + 1) as $t) == u16::MAX / 2);
            sh_test!(shr(u32::MAX, (u32::BITS + 1) as $t) == u32::MAX / 2);
            sh_test!(shr(u64::MAX, (u64::BITS + 1) as $t) == u64::MAX / 2);
            sh_test!(shr(usize::MAX, (usize::BITS + 1) as $t) == usize::MAX / 2);
        }
    }
    macro_rules! sh_test_negative_all {
        ($t:ty) => {
            sh_test!(shr(i8::MAX, -((i8::BITS + 1) as $t)) == -2);
            sh_test!(shr(i16::MAX, -((i16::BITS + 1) as $t)) == -2);
            sh_test!(shr(i32::MAX, -((i32::BITS + 1) as $t)) == -2);
            sh_test!(shr(i64::MAX, -((i64::BITS + 1) as $t)) == -2);
            sh_test!(shr(isize::MAX, -((isize::BITS + 1) as $t)) == -2);

            sh_test!(shr(u8::MAX, -((u8::BITS + 1) as $t)) == u8::MAX - 1);
            sh_test!(shr(u16::MAX, -((u16::BITS + 1) as $t)) == u16::MAX - 1);
            sh_test!(shr(u32::MAX, -((u32::BITS + 1) as $t)) == u32::MAX - 1);
            sh_test!(shr(u64::MAX, -((u64::BITS + 1) as $t)) == u64::MAX - 1);
            sh_test!(shr(usize::MAX, -((usize::BITS + 1) as $t)) == usize::MAX - 1);


            sh_test!(shl(i8::MAX, -((i8::BITS + 1) as $t)) == i8::MAX / 2);
            sh_test!(shl(i16::MAX, -((i16::BITS + 1) as $t)) == i16::MAX / 2);
            sh_test!(shl(i32::MAX, -((i32::BITS + 1) as $t)) == i32::MAX / 2);
            sh_test!(shl(i64::MAX, -((i64::BITS + 1) as $t)) == i64::MAX / 2);
            sh_test!(shl(isize::MAX, -((isize::BITS + 1) as $t)) == isize::MAX / 2);

            sh_test!(shl(u8::MAX, -((u8::BITS + 1) as $t)) == u8::MAX / 2);
            sh_test!(shl(u16::MAX, -((u16::BITS + 1) as $t)) == u16::MAX / 2);
            sh_test!(shl(u32::MAX, -((u32::BITS + 1) as $t)) == u32::MAX / 2);
            sh_test!(shl(u64::MAX, -((u64::BITS + 1) as $t)) == u64::MAX / 2);
            sh_test!(shl(usize::MAX, -((usize::BITS + 1) as $t)) == usize::MAX / 2);
        }
    }
    // FIXME(#23545): Uncomment the remaining tests
    //sh_test_all!(i8);
    //sh_test_all!(u8);
    //sh_test_all!(i16);
    //sh_test_all!(u16);
    //sh_test_all!(i32);
    //sh_test_all!(u32);
    //sh_test_all!(i64);
    //sh_test_all!(u64);
    //sh_test_all!(isize);
    sh_test_all!(usize);

    //sh_test_negative_all!(i8);
    //sh_test_negative_all!(i16);
    //sh_test_negative_all!(i32);
    //sh_test_negative_all!(i64);
    //sh_test_negative_all!(isize);
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
    macro_rules! sh_assign_test_all {
        ($t:ty) => {
            sh_assign_test!(shl_assign(i8::MAX, (i8::BITS + 1) as $t) == -2);
            sh_assign_test!(shl_assign(i16::MAX, (i16::BITS + 1) as $t) == -2);
            sh_assign_test!(shl_assign(i32::MAX, (i32::BITS + 1) as $t) == -2);
            sh_assign_test!(shl_assign(i64::MAX, (i64::BITS + 1) as $t) == -2);
            sh_assign_test!(shl_assign(isize::MAX, (isize::BITS + 1) as $t) == -2);

            sh_assign_test!(shl_assign(u8::MAX, (u8::BITS + 1) as $t) == u8::MAX - 1);
            sh_assign_test!(shl_assign(u16::MAX, (u16::BITS + 1) as $t) == u16::MAX - 1);
            sh_assign_test!(shl_assign(u32::MAX, (u32::BITS + 1) as $t) == u32::MAX - 1);
            sh_assign_test!(shl_assign(u64::MAX, (u64::BITS + 1) as $t) == u64::MAX - 1);
            sh_assign_test!(shl_assign(usize::MAX, (usize::BITS + 1) as $t) == usize::MAX - 1);


            sh_assign_test!(shr_assign(i8::MAX, (i8::BITS + 1) as $t) == i8::MAX / 2);
            sh_assign_test!(shr_assign(i16::MAX, (i16::BITS + 1) as $t) == i16::MAX / 2);
            sh_assign_test!(shr_assign(i32::MAX, (i32::BITS + 1) as $t) == i32::MAX / 2);
            sh_assign_test!(shr_assign(i64::MAX, (i64::BITS + 1) as $t) == i64::MAX / 2);
            sh_assign_test!(shr_assign(isize::MAX, (isize::BITS + 1) as $t) == isize::MAX / 2);

            sh_assign_test!(shr_assign(u8::MAX, (u8::BITS + 1) as $t) == u8::MAX / 2);
            sh_assign_test!(shr_assign(u16::MAX, (u16::BITS + 1) as $t) == u16::MAX / 2);
            sh_assign_test!(shr_assign(u32::MAX, (u32::BITS + 1) as $t) == u32::MAX / 2);
            sh_assign_test!(shr_assign(u64::MAX, (u64::BITS + 1) as $t) == u64::MAX / 2);
            sh_assign_test!(shr_assign(usize::MAX, (usize::BITS + 1) as $t) == usize::MAX / 2);
        }
    }
    macro_rules! sh_assign_test_negative_all {
        ($t:ty) => {
            sh_assign_test!(shr_assign(i8::MAX, -((i8::BITS + 1) as $t)) == -2);
            sh_assign_test!(shr_assign(i16::MAX, -((i16::BITS + 1) as $t)) == -2);
            sh_assign_test!(shr_assign(i32::MAX, -((i32::BITS + 1) as $t)) == -2);
            sh_assign_test!(shr_assign(i64::MAX, -((i64::BITS + 1) as $t)) == -2);
            sh_assign_test!(shr_assign(isize::MAX, -((isize::BITS + 1) as $t)) == -2);

            sh_assign_test!(shr_assign(u8::MAX, -((u8::BITS + 1) as $t)) == u8::MAX - 1);
            sh_assign_test!(shr_assign(u16::MAX, -((u16::BITS + 1) as $t)) == u16::MAX - 1);
            sh_assign_test!(shr_assign(u32::MAX, -((u32::BITS + 1) as $t)) == u32::MAX - 1);
            sh_assign_test!(shr_assign(u64::MAX, -((u64::BITS + 1) as $t)) == u64::MAX - 1);
            sh_assign_test!(shr_assign(usize::MAX, -((usize::BITS + 1) as $t)) == usize::MAX - 1);


            sh_assign_test!(shl_assign(i8::MAX, -((i8::BITS + 1) as $t)) == i8::MAX / 2);
            sh_assign_test!(shl_assign(i16::MAX, -((i16::BITS + 1) as $t)) == i16::MAX / 2);
            sh_assign_test!(shl_assign(i32::MAX, -((i32::BITS + 1) as $t)) == i32::MAX / 2);
            sh_assign_test!(shl_assign(i64::MAX, -((i64::BITS + 1) as $t)) == i64::MAX / 2);
            sh_assign_test!(shl_assign(isize::MAX, -((isize::BITS + 1) as $t)) == isize::MAX / 2);

            sh_assign_test!(shl_assign(u8::MAX, -((u8::BITS + 1) as $t)) == u8::MAX / 2);
            sh_assign_test!(shl_assign(u16::MAX, -((u16::BITS + 1) as $t)) == u16::MAX / 2);
            sh_assign_test!(shl_assign(u32::MAX, -((u32::BITS + 1) as $t)) == u32::MAX / 2);
            sh_assign_test!(shl_assign(u64::MAX, -((u64::BITS + 1) as $t)) == u64::MAX / 2);
            sh_assign_test!(shl_assign(usize::MAX, -((usize::BITS + 1) as $t)) == usize::MAX / 2);
        }
    }

    // FIXME(#23545): Uncomment the remaining tests
    //sh_assign_test_all!(i8);
    //sh_assign_test_all!(u8);
    //sh_assign_test_all!(i16);
    //sh_assign_test_all!(u16);
    //sh_assign_test_all!(i32);
    //sh_assign_test_all!(u32);
    //sh_assign_test_all!(i64);
    //sh_assign_test_all!(u64);
    //sh_assign_test_all!(isize);
    sh_assign_test_all!(usize);

    //sh_assign_test_negative_all!(i8);
    //sh_assign_test_negative_all!(i16);
    //sh_assign_test_negative_all!(i32);
    //sh_assign_test_negative_all!(i64);
    //sh_assign_test_negative_all!(isize);
}
