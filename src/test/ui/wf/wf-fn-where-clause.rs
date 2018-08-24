// Test that we check where-clauses on fn items.

#![feature(associated_type_defaults)]
#![feature(rustc_attrs)]
#![allow(dead_code)]

trait ExtraCopy<T:Copy> { }

fn foo<T,U>() where T: ExtraCopy<U> //~ ERROR E0277
{
}

#[rustc_error]
fn main() { }
