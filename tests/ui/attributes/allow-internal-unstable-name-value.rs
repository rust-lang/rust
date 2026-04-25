//@ check-pass

#![feature(allow_internal_unstable)]
#![allow(internal_features)]

#[allow_internal_unstable(cat = "meow")]
macro_rules! foo {
    () => {}
}

fn main() {}
