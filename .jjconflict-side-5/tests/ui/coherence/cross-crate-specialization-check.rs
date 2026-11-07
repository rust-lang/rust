//! Regression test for <https://github.com/rust-lang/rust/issues/37291>.
//!
//! The problem was that the starting environment for a specialization
//! check was not including the where-clauses from the impl when attempting
//! to normalize the impl's trait-ref, so things like `<C as Foo>::Item`
//! could not resolve, since the `C: Foo` trait bound was not included in
//! the environment.

//@ aux-build:cross-crate-specialization-check.rs
//@ run-pass

#![allow(unused_imports)]

extern crate cross_crate_specialization_check;

use cross_crate_specialization_check::{CV, WrapperB, WrapperC};

fn main() {
    let a = WrapperC(CV);
    let b = WrapperC(CV);
    if false {
        let _ = a * b;
    }
}
