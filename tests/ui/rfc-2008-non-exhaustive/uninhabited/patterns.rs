// aux-build:uninhabited.rs
// build-pass (FIXME(62277): could be check-pass?)
#![deny(unreachable_patterns)]
#![feature(exhaustive_patterns)]

extern crate uninhabited;

use uninhabited::{
    PartiallyInhabitedVariants,
    UninhabitedEnum,
    UninhabitedStruct,
    UninhabitedTupleStruct,
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

// This test checks that non-exhaustive types that would normally be considered uninhabited within
// the defining crate are not considered uninhabited from extern crates.

fn main() {
    match uninhabited_enum() {
        Some(_x) => (), // This line would normally error.
        None => (),
    }

    match uninhabited_variant() {
        Some(_x) => (), // This line would normally error.
        None => (),
    }

    // This line would normally error.
    while let PartiallyInhabitedVariants::Struct { x, .. } = partially_inhabited_variant() {
    }

    while let Some(_x) = uninhabited_struct() { // This line would normally error.
    }

    while let Some(_x) = uninhabited_tuple_struct() { // This line would normally error.
    }
}
