// aux-build:uninhabited.rs
#![deny(unreachable_patterns)]
#![feature(exhaustive_patterns)]
#![feature(never_type)]

extern crate uninhabited;

use uninhabited::{
    UninhabitedEnum,
    UninhabitedStruct,
    UninhabitedTupleStruct,
    UninhabitedVariants,
};

struct A;

// This test checks that an empty match on a non-exhaustive uninhabited type from an extern crate
// will not compile. In particular, this enables the `exhaustive_patterns` feature as this can
// change the branch used in the compiler to determine this.

fn cannot_empty_match_on_empty_enum_to_anything(x: UninhabitedEnum) -> A {
    match x {} //~ ERROR non-exhaustive patterns
}

fn cannot_empty_match_on_empty_struct_to_anything(x: UninhabitedStruct) -> A {
    match x {} //~ ERROR non-exhaustive patterns
}

fn cannot_empty_match_on_empty_tuple_struct_to_anything(x: UninhabitedTupleStruct) -> A {
    match x {} //~ ERROR non-exhaustive patterns
}

fn cannot_empty_match_on_enum_with_empty_variants_struct_to_anything(x: UninhabitedVariants) -> A {
    match x {} //~ ERROR non-exhaustive patterns
}

fn main() {}
