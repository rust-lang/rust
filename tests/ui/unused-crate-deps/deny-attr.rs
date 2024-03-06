// Check for unused crate dep, no path

//@ edition:2018
//@ aux-crate:bar=bar.rs

#![deny(unused_crate_dependencies)]
//~^ ERROR external crate `bar` unused in

fn main() {}
