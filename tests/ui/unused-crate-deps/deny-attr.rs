// Check for unused crate dep, no path

//@ edition:2018
//@ aux-crate:bar=bar.rs

#![deny(unused_crate_dependencies)]
//~^ ERROR extern crate `bar` is unused in

fn main() {}
