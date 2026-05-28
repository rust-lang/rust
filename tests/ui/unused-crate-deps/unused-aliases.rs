// Warn about unused aliased for the crate

//@ edition:2018
//@ check-pass
//@ aux-crate:bar=bar.rs
//@ aux-crate:barbar=bar.rs

#![warn(unused_crate_dependencies)]
//~^ WARNING extern crate `barbar` is unused in

use bar as _;

fn main() {}
