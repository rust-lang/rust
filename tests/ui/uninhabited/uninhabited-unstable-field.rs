//@ aux-build: staged-api.rs
//@ revisions: current exhaustive
#![cfg_attr(exhaustive, feature(exhaustive_patterns))]
#![feature(never_type)]
#![feature(my_coro_state)] // Custom feature from `staged-api.rs`
#![deny(unreachable_patterns)]

extern crate staged_api;

use staged_api::{Foo, MyCoroutineState};

enum Void {}

fn demo(x: Foo<Void>) {
    match x {}
    //~^ ERROR non-exhaustive patterns
}

// Ensure that the pattern is not considered unreachable.
fn demo2(x: Foo<Void>) {
    match x {
        Foo { .. } => {}
    }
}

// Same as above, but for wildcard.
fn demo3(x: Foo<Void>) {
    match x {
        _ => {}
    }
}

fn unstable_enum(x: MyCoroutineState<i32, !>) {
    match x {
        //~^ ERROR non-exhaustive patterns
        MyCoroutineState::Yielded(_) => {}
    }
    match x {
        MyCoroutineState::Yielded(_) => {}
        MyCoroutineState::Complete(_) => {}
    }
}

fn main() {}
