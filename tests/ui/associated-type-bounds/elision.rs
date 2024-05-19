#![feature(anonymous_lifetime_in_impl_trait)]

// The same thing should happen for constraints in dyn trait.
fn f(x: &mut dyn Iterator<Item: Iterator<Item = &'_ ()>>) -> Option<&'_ ()> { x.next() }
//~^ ERROR associated type bounds are not allowed in `dyn` types
//~| ERROR missing lifetime specifier

fn main() {}
