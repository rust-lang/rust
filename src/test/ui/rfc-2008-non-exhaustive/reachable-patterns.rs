// Test that the reachable_patterns lint is triggered correctly.

#![deny(reachable_patterns)]

// aux-build:enums.rs
extern crate enums;
// aux-build:structs.rs
extern crate structs;

use enums::{EmptyNonExhaustiveEnum, NonExhaustiveEnum};
use structs::{NormalStruct, UnitStruct, TupleStruct, FunctionalRecord};

#[non_exhaustive]
pub struct Foo {
    a: u8,
    b: usize,
    c: String,
}

#[non_exhaustive]
pub enum Bar {
    A,
    B,
    C,
}

fn main() {
    let enumeration = Bar::A;
    let structure = Foo { a: 1, b: 1, c: "hello".to_owned(), };

    match enumeration {
        Bar::A => {},
        Bar::B => {},
        #[deny(reachable)]
        _ => {}
    }
    //~^^^^^^ missing patterns of non_exhaustive type
    //~| HELP add `Bar::C` to match all reachable patterns

    match enumeration {
        Bar::A => {},
        Bar::B => {},
        _ => {}
    }

    match NonExhaustiveEnum::Unit {
        NonExhaustiveEnum::Unit => {},
        NonExhaustiveEnum::Tuple(_) => {},
        #[warn(reachable)]
        _ => {}
    }
    //~^^^^^^ missing patterns of non_exhaustive type
    //~| HELP add `Struct` to match all reachable patterns

    let Foo { a, b, .. } = structure;
    //~^ ERROR missing patterns of non_exhaustive type
    //~| HELP add `c` to match all reachable patterns

    let FunctionalRecord { first_field, second_field, .. } = FunctionalRecord::default();
    //~^ ERROR missing patterns of non_exhaustive type
    //~| HELP add `third_field` to match all reachable patterns


}
