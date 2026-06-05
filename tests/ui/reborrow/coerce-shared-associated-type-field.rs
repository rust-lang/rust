//@ check-pass

#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, Reborrow};

trait Trait {
    type Assoc;
}

impl Trait for i32 {
    type Assoc = i64;
}

struct MyMut<'a> {
    x: &'a (),
    y: i64,
}

#[derive(Copy, Clone)]
struct MyRef<'a> {
    x: &'a (),
    y: <i32 as Trait>::Assoc,
}

impl Reborrow for MyMut<'_> {}

impl<'a> CoerceShared<MyRef<'a>> for MyMut<'a> {}

fn main() {}
