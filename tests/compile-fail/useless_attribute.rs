#![feature(plugin)]
#![plugin(clippy)]
#![deny(useless_attribute)]

#[allow(dead_code)] //~ ERROR useless lint attribute
//~| HELP if you just forgot a `!`, use
//~| SUGGESTION #![allow(dead_code)]
extern crate clippy_lints;

// don't lint on unused_import for `use` items
#[allow(unused_imports)]
use std::collections;

fn main() {}
