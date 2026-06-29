//@ run-pass
#![allow(dead_code)]
//@ aux-build:extern-trait-bound-with-array-associated-type.rs
//! Regression test for https://github.com/rust-lang/rust/issues/48984
//! This test exposes an error that occurs when a base trait has at
//! least one associated type, and the extending trait specifies the
//! associated type as an array of some kind. In order to trigger the
//! error, the extending trait must be defined in an external crate, and
//! used to constrain a generic type parameter.
extern crate extern_trait_bound_with_array_associated_type as issue48984aux;
use issue48984aux::Bar;

fn do_thing<T: Bar>() { }

fn main() { }
