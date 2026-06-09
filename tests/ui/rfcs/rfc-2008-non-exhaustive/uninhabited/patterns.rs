//@ aux-build:uninhabited.rs
#![feature(exhaustive_patterns)]
#![deny(unreachable_patterns)]

extern crate uninhabited;

use uninhabited::{
    PartiallyInhabitedVariants, UninhabitedEnum, UninhabitedStruct, UninhabitedTupleStruct,
    UninhabitedVariants,
};

fn uninhabited_enum() -> Option<UninhabitedEnum> {
    None
}

fn uninhabited_variant() -> Option<UninhabitedVariants> {
    None
}

fn partially_inhabited_variant() -> PartiallyInhabitedVariants {
    PartiallyInhabitedVariants::Tuple(3)
}

fn uninhabited_struct() -> Option<UninhabitedStruct> {
    None
}

fn uninhabited_tuple_struct() -> Option<UninhabitedTupleStruct> {
    None
}

// This test checks that non-exhaustive enums are never considered uninhabited outside their
// defining crate, and non-exhaustive structs are considered uninhabited the same way as normal
// ones.
fn main() {
    match uninhabited_enum() {
        Some(_x) => (), // This would error without `non_exhaustive`
        None => (),
    }

    match uninhabited_variant() {
        Some(_x) => (), //~ ERROR unreachable
        None => (),
    }

    // This line would normally error.
    while let PartiallyInhabitedVariants::Struct { x, .. } = partially_inhabited_variant() {} //~ ERROR unreachable

    while let Some(_x) = uninhabited_struct() { //~ ERROR unreachable
    }

    while let Some(_x) = uninhabited_tuple_struct() { //~ ERROR unreachable
    }
}
