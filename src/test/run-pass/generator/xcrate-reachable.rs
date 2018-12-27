// run-pass

// aux-build:xcrate-reachable.rs

#![feature(generator_trait)]

extern crate xcrate_reachable as foo;

use std::ops::Generator;

fn main() {
    unsafe { foo::foo().resume(); }
}
