//@ check-pass
// Regression test for <https://github.com/rust-lang/rust/issues/157313>

#![feature(reborrow)]

use std::marker::Reborrow;

struct MyMut<'a>(*mut &'a ());

impl Reborrow for MyMut<'_> {}

fn foo(x: MyMut) {
    let _y: MyMut = x;
}

fn main() {}
