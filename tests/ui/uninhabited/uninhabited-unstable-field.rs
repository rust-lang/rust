//@ aux-build: staged-api.rs
//! The field of `Pin` used to be public, which would cause `Pin<Void>` to be uninhabited. To remedy
//! this, we temporarily made it so unstable fields are always considered inhabited. This has now
//! been reverted, and this file ensures that we don't special-case unstable fields wrt
//! inhabitedness anymore.
#![feature(exhaustive_patterns)]
#![feature(never_type)]
#![feature(my_coro_state)] // Custom feature from `staged-api.rs`
#![deny(unreachable_patterns)]

extern crate staged_api;

use staged_api::{Foo, MyCoroutineState};

enum Void {}

fn demo(x: Foo<Void>) {
    match x {}
}

// Ensure that the pattern is considered unreachable.
fn demo2(x: Foo<Void>) {
    match x {
        Foo { .. } => {} //~ ERROR unreachable
    }
}

// Same as above, but for wildcard.
fn demo3(x: Foo<Void>) {
    match x {
        _ => {} //~ ERROR unreachable
    }
}

fn unstable_enum(x: MyCoroutineState<i32, !>) {
    match x {
        MyCoroutineState::Yielded(_) => {}
    }
    match x {
        MyCoroutineState::Yielded(_) => {}
        MyCoroutineState::Complete(_) => {} //~ ERROR unreachable
    }
}

fn main() {}
