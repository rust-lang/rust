//@ check-pass

//! Test that CoerceShared in const contexts can convert a 'static source into a 'static target.

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

struct MyMut<'a>(&'a u8);
impl Reborrow for MyMut<'_> {}

#[derive(Clone, Copy)]
struct MyRef<'a>(&'a u8);
impl<'a> CoerceShared<MyRef<'a>> for MyMut<'a> {}

const fn coerce(x: MyRef<'_>) -> MyRef<'_> {
    x
}

static BAD: &'static MyRef<'static> = &coerce(MyMut(&1));

fn main() {}
