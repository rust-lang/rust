//@ aux-build: staged-api.rs
//@ revisions: current exhaustive

#![feature(exhaustive_patterns)]

extern crate staged_api;

use staged_api::Foo;

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
    match x { _ => {} }
}

fn main() {}
