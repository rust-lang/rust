// aux-build:uninhabited.rs
#![deny(unreachable_patterns)]
#![feature(never_type)]
#![feature(non_exhaustive)]

extern crate uninhabited;

use uninhabited::PartiallyInhabitedVariants;

// This test checks a redundant/useless pattern of a non-exhaustive enum/variant is still
// warned against.

pub fn foo(x: PartiallyInhabitedVariants) {
    match x {
        PartiallyInhabitedVariants::Struct { .. } => {},
        PartiallyInhabitedVariants::Struct { .. } => {},
        //~^ ERROR unreachable pattern
        _ => {},
    }
}

fn main() { }
