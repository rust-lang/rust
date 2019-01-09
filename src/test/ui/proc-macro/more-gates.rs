// aux-build:more-gates.rs

#![feature(decl_macro)]

extern crate more_gates as foo;

use foo::*;

#[attr2mac1]
//~^ ERROR: cannot expand to macro definitions
pub fn a() {}
#[attr2mac2]
//~^ ERROR: cannot expand to macro definitions
pub fn a() {}

mac2mac1!(); //~ ERROR: cannot expand to macro definitions
mac2mac2!(); //~ ERROR: cannot expand to macro definitions

tricky!();
//~^ ERROR: cannot expand to macro definitions

fn main() {}
