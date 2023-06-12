// check-pass

#![deny(unreachable_patterns)]
#![feature(exhaustive_patterns)]
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

pub struct IndirectUninhabitedEnum(UninhabitedEnum);

pub struct IndirectUninhabitedStruct(UninhabitedStruct);

pub struct IndirectUninhabitedTupleStruct(UninhabitedTupleStruct);

pub struct IndirectUninhabitedVariants(UninhabitedVariants);

struct A;

// This test checks that an empty match on a non-exhaustive uninhabited type from the defining crate
// will compile. In particular, this enables the `exhaustive_patterns` feature as this can
// change the branch used in the compiler to determine this.
// Codegen is skipped because tests with long names can cause issues on Windows CI, see #60648.

fn cannot_empty_match_on_empty_enum_to_anything(x: IndirectUninhabitedEnum) -> A {
    match x {}
}

fn cannot_empty_match_on_empty_struct_to_anything(x: IndirectUninhabitedStruct) -> A {
    match x {}
}

fn cannot_empty_match_on_empty_tuple_struct_to_anything(x: IndirectUninhabitedTupleStruct) -> A {
    match x {}
}

fn cannot_empty_match_on_enum_with_empty_variants_struct_to_anything(
    x: IndirectUninhabitedVariants,
) -> A {
    match x {}
}

fn main() {}
