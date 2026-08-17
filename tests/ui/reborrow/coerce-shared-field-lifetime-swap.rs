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
    //~^ ERROR
    y: &'static (),
}

impl<'a> CoerceShared<MyRef<'a>> for MyMut<'a> {}

fn main() {}
