// compile-flags: -Z print-type-sizes
// build-pass
// ignore-pass
// ^-- needed because `--pass check` does not emit the output needed.
//     FIXME: consider using an attribute instead of side-effects.

// This file illustrates how niche-filling enums are handled,
// modelled after cases like `Option<&u32>`, `Option<bool>` and such.
//
// It uses NonZeroU32 rather than `&_` or `Unique<_>`, because
// the test is not set up to deal with target-dependent pointer width.
//
// It avoids using u64/i64 because on some targets that is only 4-byte
// aligned (while on most it is 8-byte aligned) and so the resulting
// padding and overall computed sizes can be quite different.

#![feature(start)]
#![allow(dead_code)]

use std::num::NonZeroU32;

pub enum MyOption<T> { None, Some(T) }

impl<T> Default for MyOption<T> {
    fn default() -> Self { MyOption::None }
}

pub enum EmbeddedDiscr {
    None,
    Record { pre: u8, val: NonZeroU32, post: u16 },
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
    val: NonZeroU32,
    post: u16,
}

impl Default for NestedNonZero {
    fn default() -> Self {
        NestedNonZero { pre: 0, val: NonZeroU32::new(1).unwrap(), post: 0 }
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

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let _x: MyOption<NonZeroU32> = Default::default();
    let _y: EmbeddedDiscr = Default::default();
    let _z: MyOption<IndirectNonZero> = Default::default();
    let _a: MyOption<bool> = Default::default();
    let _b: MyOption<char> = Default::default();
    let _c: MyOption<std::cmp::Ordering> = Default::default();
    let _b: MyOption<MyOption<u8>> = Default::default();
    let _e: Enum4<(), char, (), ()> = Enum4::One(());
    let _f: Enum4<(), (), bool, ()> = Enum4::One(());
    let _g: Enum4<(), (), (), MyOption<u8>> = Enum4::One(());

    // Unions do not currently participate in niche filling.
    let _h: MyOption<Union2<NonZeroU32, u32>> = Default::default();

    // ...even when theoretically possible.
    let _i: MyOption<Union1<NonZeroU32>> = Default::default();
    let _j: MyOption<Union2<NonZeroU32, NonZeroU32>> = Default::default();

    0
}
