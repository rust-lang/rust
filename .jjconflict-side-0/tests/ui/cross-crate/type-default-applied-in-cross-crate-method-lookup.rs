//! Regression test for https://github.com/rust-lang/rust/issues/30123
//@ aux-build:type-default-applied-in-cross-crate-method-lookup.rs

extern crate type_default_applied_in_cross_crate_method_lookup;
use type_default_applied_in_cross_crate_method_lookup::*;

fn main() {
    let ug = Graph::<i32, i32>::new_undirected();
    //~^ ERROR no associated function or constant named `new_undirected` found
}
