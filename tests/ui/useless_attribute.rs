#![feature(plugin)]
#![plugin(clippy)]
#![deny(useless_attribute)]

#[allow(dead_code)]
extern crate clippy_lints;

// don't lint on unused_import for `use` items
#[allow(unused_imports)]
use std::collections;

// don't lint on deprecated for `use` items
mod foo { #[deprecated] pub struct Bar; }
#[allow(deprecated)]
pub use foo::Bar;

fn main() {}
