//@ run-pass
#![allow(unused_imports)]
//@ aux-build:lib.rs

// Regression test for #37291. The problem was that the starting
// environment for a specialization check was not including the
// where-clauses from the impl when attempting to normalize the impl's
// trait-ref, so things like `<C as Foo>::Item` could not resolve,
// since the `C: Foo` trait bound was not included in the environment.

extern crate lib;

use lib::{CV, WrapperB, WrapperC};

fn main() {
    let a = WrapperC(CV);
    let b = WrapperC(CV);
    if false {
        let _ = a * b;
    }
}
