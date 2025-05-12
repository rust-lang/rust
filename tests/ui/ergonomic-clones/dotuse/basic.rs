//@ check-pass

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::clone::UseCloned;

fn basic_test(x: i32) -> i32 {
    x.use.use.abs()
}

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}

fn do_not_move_test(x: Foo) -> Foo {
    let s = x.use;
    x
}

fn main() {}
