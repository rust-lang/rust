//@ check-fail

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
//~^ ERROR temporary value dropped while borrowed

fn main() {}
