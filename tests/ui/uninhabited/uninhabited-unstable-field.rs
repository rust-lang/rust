//@ aux-build: staged-api.rs

extern crate staged_api;

use staged_api::Foo;

enum Void {}

fn demo(x: Foo<Void>) {
    match x {}
    //~^ ERROR non-exhaustive patterns
}

fn main() {}
