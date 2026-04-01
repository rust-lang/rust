//! This file illustrates how niche-filling enums are handled,
//! modelled after cases like `Option<&u32>`, `Option<bool>` and such.
//!
//! It uses `NonZero<u32>` rather than `&_` or `Unique<_>`, because
//! the test is not set up to deal with target-dependent pointer width.
//!
//! It avoids using `u64`/`i64` because on some targets that is only 4-byte
//! aligned (while on most it is 8-byte aligned) and so the resulting
//! padding and overall computed sizes can be quite different.
//!
//@ compile-flags: -Z print-type-sizes --crate-type lib
//@ ignore-std-debug-assertions (debug assertions will print more types)
//@ build-pass
//@ ignore-pass
//  ^-- needed because `--pass check` does not emit the output needed.
//      FIXME: consider using an attribute instead of side-effects.
#![allow(dead_code)]
#![feature(rustc_attrs)]

use std::num::NonZero;

pub enum MyOption<T> { None, Some(T) }

#[rustc_layout_scalar_valid_range_start(0)]
#[rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FE)]
pub struct MyNotNegativeOne {
  _i: i32,
}

impl<T> Default for MyOption<T> {
    fn default() -> Self { MyOption::None }
}

pub enum EmbeddedDiscr {
    None,
    Record { pre: u8, val: NonZero<u32>, post: u16 },
}

impl Default for EmbeddedDiscr {
    fn default() -> Self { EmbeddedDiscr::None }
}

#[derive(Default)]
pub struct IndirectNonZero {
    pre: u8,
    nested: NestedNonZero,
    post: u16,
}

pub struct NestedNonZero {
    pre: u8,
    val: NonZero<u32>,
    post: u16,
}

impl Default for NestedNonZero {
    fn default() -> Self {
        // Ideally we'd call NonZero::new_unchecked, but this test is supposed
        // to be target-independent and NonZero::new_unchecked is #[track_caller]
        // (see #129658) so mentioning that function pulls in std::panic::Location
        // which contains a &str, whose layout is target-dependent.
        const ONE: NonZero<u32> = const {
            unsafe { std::mem::transmute(1u32) }
        };
        NestedNonZero { pre: 0, val: ONE, post: 0 }
    }
}

pub enum Enum4<A, B, C, D> {
    One(A),
    Two(B),
    Three(C),
    Four(D)
}

pub union Union1<A: Copy> {
    a: A,
}

pub union Union2<A: Copy, B: Copy> {
    a: A,
    b: B,
}

pub fn test() {
    let _x: MyOption<NonZero<u32>> = Default::default();
    let _y: EmbeddedDiscr = Default::default();
    let _z: MyOption<IndirectNonZero> = Default::default();
    let _a: MyOption<bool> = Default::default();
    let _b: MyOption<char> = Default::default();
    let _c: MyOption<std::cmp::Ordering> = Default::default();
    let _d: MyOption<MyOption<u8>> = Default::default();
    let _e: Enum4<(), char, (), ()> = Enum4::One(());
    let _f: Enum4<(), (), bool, ()> = Enum4::One(());
    let _g: Enum4<(), (), (), MyOption<u8>> = Enum4::One(());
    let _h: MyOption<MyNotNegativeOne> = Default::default();

    // Unions do not currently participate in niche filling.
    let _i: MyOption<Union2<NonZero<u32>, u32>> = Default::default();

    // ...even when theoretically possible.
    let _j: MyOption<Union1<NonZero<u32>>> = Default::default();
    let _k: MyOption<Union2<NonZero<u32>, NonZero<u32>>> = Default::default();
}
