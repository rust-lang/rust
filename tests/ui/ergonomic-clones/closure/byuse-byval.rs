//@ run-pass

#![allow(incomplete_features)]
#![feature(ergonomic_clones)]

use std::clone::UseCloned;

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}

fn do_not_move_test(x: Foo) -> Foo {
    let cl = (|| {
        let _s = x.use;
        x
    });

    cl()
}

fn main() {
    do_not_move_test(Foo);
}
