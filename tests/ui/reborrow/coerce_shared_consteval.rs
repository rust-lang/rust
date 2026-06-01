//@ run-pass
// Regression test for the const-eval reproducer in rust-lang/rust#156313.

#![feature(reborrow)]
#![allow(dead_code)]
#![allow(unused_variables)]

use std::marker::{CoerceShared, Reborrow};

pub struct MyMut<'a>(&'a mut u8);

impl Reborrow for MyMut<'_> {}

#[derive(Clone, Copy)]
pub struct MyRef<'a>(&'a u8);

impl<'a> CoerceShared<MyRef<'a>> for MyMut<'a> {}

const fn consteval_reproducer() {
    let mut value = 1;
    foo(MyMut(&mut value));
}

const fn foo(_x: MyRef<'_>) {}

fn main() {
    const { consteval_reproducer(); }
}
