#![feature(never_type)]
#![feature(non_exhaustive)]

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

pub struct IndirectUninhabitedEnum(UninhabitedEnum);

pub struct IndirectUninhabitedStruct(UninhabitedStruct);

pub struct IndirectUninhabitedTupleStruct(UninhabitedTupleStruct);

pub struct IndirectUninhabitedVariants(UninhabitedVariants);

struct A;

// This test checks that an empty match on a non-exhaustive uninhabited type through a level of
// indirection from the defining crate will not compile without `#![feature(exhaustive_patterns)]`.

fn cannot_empty_match_on_empty_enum_to_anything(x: IndirectUninhabitedEnum) -> A {
    match x {} //~ ERROR non-exhaustive patterns
}

fn cannot_empty_match_on_empty_struct_to_anything(x: IndirectUninhabitedStruct) -> A {
    match x {} //~ ERROR non-exhaustive patterns
}

fn cannot_empty_match_on_empty_tuple_struct_to_anything(x: IndirectUninhabitedTupleStruct) -> A {
    match x {} //~ ERROR non-exhaustive patterns
}

fn cannot_empty_match_on_enum_with_empty_variants_struct_to_anything(
    x: IndirectUninhabitedVariants,
) -> A {
    match x {} //~ ERROR non-exhaustive patterns
}

fn main() {}
