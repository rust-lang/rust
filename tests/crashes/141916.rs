//@ known-bug: rust-lang/rust#141916
//@ edition: 2024
#![allow(incomplete_features)]
#![feature(ergonomic_clones)]

use std::clone::UseCloned;

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}

fn do_not_move_test(x: Foo) -> Foo { async {
    let s = x.use;
    x
} }

fn main() {}
