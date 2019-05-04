// aux-build:uninhabited.rs
#![feature(never_type)]

extern crate uninhabited;

use uninhabited::{
    UninhabitedEnum,
};

struct A;

// This test checks that an empty match on a non-exhaustive uninhabited type from an extern crate
// will not compile.

fn cannot_empty_match_on_empty_enum_to_anything(x: UninhabitedEnum) -> A {
    match x {} //~ ERROR non-exhaustive patterns
}

fn main() {}
