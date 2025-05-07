//@ run-pass
//@ check-run-results
#![feature(dyn_star, pointer_like_trait)]
#![allow(incomplete_features)]

use std::fmt::Debug;
use std::marker::PointerLike;

#[derive(Debug)]
#[repr(transparent)]
struct Foo(#[allow(dead_code)] usize);

// FIXME(dyn_star): Make this into a derive.
impl PointerLike for Foo {}

impl Drop for Foo {
    fn drop(&mut self) {
        println!("destructor called");
    }
}

fn make_dyn_star(i: Foo) {
    let _dyn_i: dyn* Debug = i;
}

fn main() {
    make_dyn_star(Foo(42));
}
