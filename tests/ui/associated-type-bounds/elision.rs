#![feature(associated_type_bounds)]

// The same thing should happen for constraints in dyn trait.
fn f(x: &mut dyn Iterator<Item: Iterator<Item = &'_ ()>>) -> Option<&'_ ()> { x.next() }
//~^ ERROR missing lifetime specifier
//~| ERROR mismatched types

fn main() {}
