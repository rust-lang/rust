#![feature(never_type)]

#[non_exhaustive]
pub enum UninhabitedEnum {
}

#[non_exhaustive]
pub struct UninhabitedStruct {
    _priv: !,
}

#[non_exhaustive]
pub struct UninhabitedTupleStruct(!);

pub enum UninhabitedVariants {
    #[non_exhaustive] Tuple(!),
    #[non_exhaustive] Struct { x: ! }
}

struct A;

// This test checks that an empty match on a non-exhaustive uninhabited type from the defining crate
// will compile.

fn cannot_empty_match_on_empty_enum_to_anything(x: UninhabitedEnum) -> A {
    match x {}
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
