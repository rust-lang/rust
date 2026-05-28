//@ aux-build:uninhabited.rs
#![feature(never_type)]

extern crate uninhabited;

use uninhabited::*;

struct A;

// This test checks that an empty match on a non-exhaustive uninhabited type from an extern crate
// will not compile.

fn cannot_empty_match_on_empty_enum_to_anything(x: UninhabitedEnum) -> A {
    match x {} //~ ERROR non-exhaustive patterns
}

fn empty_match_on_empty_struct(x: UninhabitedStruct) -> A {
    match x {}
}

fn cannot_empty_match_on_privately_empty_struct(x: PrivatelyUninhabitedStruct) -> A {
    match x {} //~ ERROR non-exhaustive patterns
}

fn empty_match_on_empty_tuple_struct(x: UninhabitedTupleStruct) -> A {
    match x {}
}

fn empty_match_on_enum_with_empty_variants_struct(x: UninhabitedVariants) -> A {
    match x {}
}

fn main() {}
