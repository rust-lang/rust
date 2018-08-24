// edition:2018
// aux-build:removing-extern-crate.rs
// run-rustfix
// compile-pass

#![warn(rust_2018_idioms)]
#![allow(unused_imports)]

extern crate std as foo;
extern crate core;

mod another {
    extern crate std as foo;
    extern crate std;
}

fn main() {}
