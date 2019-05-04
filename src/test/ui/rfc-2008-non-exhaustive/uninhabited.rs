// aux-build:uninhabited.rs
#![feature(never_type)]

extern crate uninhabited;

use uninhabited::{
    UninhabitedEnum,
    UninhabitedStruct,
    UninhabitedTupleStruct,
    UninhabitedVariants,
};

// This test checks that uninhabited non-exhaustive types cannot coerce to any type, as the never
// type can.

struct A;

fn can_coerce_never_type_to_anything(x: !) -> A {
    x
}

fn cannot_coerce_empty_enum_to_anything(x: UninhabitedEnum) -> A {
    x //~ ERROR mismatched types
}

fn cannot_coerce_empty_tuple_struct_to_anything(x: UninhabitedTupleStruct) -> A {
    x //~ ERROR mismatched types
}

fn cannot_coerce_empty_struct_to_anything(x: UninhabitedStruct) -> A {
    x //~ ERROR mismatched types
}

fn cannot_coerce_enum_with_empty_variants_to_anything(x: UninhabitedVariants) -> A {
    x //~ ERROR mismatched types
}

fn main() {}
