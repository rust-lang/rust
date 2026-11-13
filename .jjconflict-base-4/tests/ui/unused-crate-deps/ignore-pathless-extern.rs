// Pathless --extern references don't count

//@ edition:2018
//@ check-pass
//@ aux-crate:bar=bar.rs
//@ compile-flags:--extern proc_macro

#![warn(unused_crate_dependencies)]

use bar as _;

fn main() {}
