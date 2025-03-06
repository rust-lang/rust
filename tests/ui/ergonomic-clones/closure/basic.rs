//@ check-pass

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::clone::UseCloned;
use std::future::Future;

fn ergonomic_clone_closure_no_captures() -> i32 {
    let cl = use || {
        1
    };
    cl()
}

fn ergonomic_clone_closure_move() -> String {
    let s = String::from("hi");

    let cl = use || {
        s
    };
    cl()
}

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}

fn ergonomic_clone_closure_use_cloned() -> Foo {
    let f = Foo;

    let f1 = use || {
        f
    };

    let f2 = use || {
        f
    };

    f
}

fn ergonomic_clone_closure_copy() -> i32 {
    let i = 1;

    let i1 = use || {
        i
    };

    let i2 = use || {
        i
    };

    i
}

fn main() {}
