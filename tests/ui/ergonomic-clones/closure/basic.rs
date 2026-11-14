//@ check-pass

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::clone::UseCloned;
use std::future::Future;
use std::sync::Arc;

fn ergonomic_clone_closure_no_captures() -> i32 {
    let cl = || 1.use;
    cl()
}

fn ergonomic_clone_closure_move() -> Arc<String> {
    let s = Arc::new(String::from("hi"));

    let cl = || s.use;
    cl()
}

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}

fn ergonomic_clone_closure_use_cloned() -> Foo {
    let f = Foo;

    let f1 = || f.use;

    let f2 = || f.use;

    f
}

fn ergonomic_clone_closure_copy() -> i32 {
    let i = 1;

    let i1 = || i.use;

    let i2 = || i.use;

    i
}

fn main() {}
