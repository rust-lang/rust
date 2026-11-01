//@ run-rustfix
//@ rustfix-only-machine-applicable

// Make sure that simple overcapture suggestions remain machine applicable.

#![allow(unused)]
#![deny(impl_trait_overcaptures)]

fn named<'a>(x: &'a i32) -> impl Sized { *x }
//~^ ERROR `impl Sized` will capture more lifetimes than possibly intended in edition 2024
//~| WARN this changes meaning in Rust 2024

fn main() {}
