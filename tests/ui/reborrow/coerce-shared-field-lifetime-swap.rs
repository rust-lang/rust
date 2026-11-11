//! Test that CoerceShared cannot be used to swap 'static and 'a lifetimes around.

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

struct MyMut<'a> {
    x: &'static (),
    y: &'a (),
}

impl Reborrow for MyMut<'_> {}

#[derive(Copy, Clone)]
struct MyRef<'a> {
    x: &'a (),
    y: &'static (),
    //~^ ERROR
}

impl<'a: 'b, 'b> CoerceShared<MyRef<'b>> for MyMut<'a> {}

fn main() {}
