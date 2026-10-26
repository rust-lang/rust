//! Regression test for an ICE in borrowck when a `CoerceShared` coercion goes through an impl
//! whose source contains a `&'static mut` field.
//! See <https://github.com/rust-lang/rust/issues/162141>.

#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, Reborrow};

struct ReorderMut<'a> {
    a: &'static mut u8,
    b: &'a mut u16,
}

impl<'a> Reborrow for ReorderMut<'a> {}

#[derive(Clone, Copy)]
struct ReorderRef<'a> {
    b: &'a u16,
    a: &'a u8,
    //~^ ERROR implementing `CoerceShared` requires corresponding fields to match, be reborrowable with `CoerceShared`, or coerce a mutable reference field to a shared reference field
}

impl<'a> CoerceShared<ReorderRef<'a>> for ReorderMut<'a> {}

fn read(value: ReorderRef<'_>) -> (u16, u8) {
    (*value.b, *value.a)
}

fn main() {
    let mut a = 1;
    let mut b = 2;
    let wrapped = ReorderMut { a: &mut a, b: &mut b };

    assert_eq!(read(wrapped), (2, 1));
}
