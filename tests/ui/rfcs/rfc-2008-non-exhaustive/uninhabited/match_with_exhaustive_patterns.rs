//@ aux-build:uninhabited.rs
#![deny(unreachable_patterns)]
#![feature(never_type)]

extern crate uninhabited;

use uninhabited::{
    UninhabitedEnum, UninhabitedStruct, UninhabitedTupleStruct, UninhabitedVariants,
};

struct A;

// This test checks that non-exhaustive enums are never considered uninhabited outside their
// defining crate, and non-exhaustive structs are considered uninhabited the same way as normal
// ones.
fn cannot_empty_match_on_non_exhaustive_empty_enum(x: UninhabitedEnum) -> A {
    match x {} //~ ERROR non-exhaustive patterns
}

fn empty_match_on_empty_struct(x: UninhabitedStruct) -> A {
    match x {}
}

fn empty_match_on_empty_tuple_struct(x: UninhabitedTupleStruct) -> A {
    match x {}
}

fn empty_match_on_enum_with_empty_variants_struct(x: UninhabitedVariants) -> A {
    match x {}
}

fn main() {}
