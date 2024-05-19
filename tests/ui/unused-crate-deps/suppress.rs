// Suppress by using crate

//@ edition:2018
//@ check-pass
//@ aux-crate:bar=bar.rs

#![warn(unused_crate_dependencies)]

use bar as _;

fn main() {}
