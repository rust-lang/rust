//@ run-pass
//@ aux-build:explicit_self_xcrate.rs


extern crate explicit_self_xcrate;
use explicit_self_xcrate::{Foo, Bar};

pub fn main() {
    let x = Bar { x: "hello".to_string() };
    x.f();
}
