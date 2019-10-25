// check-pass

#![deny(unreachable_patterns)]
#![feature(exhaustive_patterns)]
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

struct A;

// This test checks that an empty match on a non-exhaustive uninhabited type from the defining crate
// will compile. In particular, this enables the `exhaustive_patterns` feature as this can
// change the branch used in the compiler to determine this.
// Codegen is skipped because tests with long names can cause issues on Windows CI, see #60648.

fn cannot_empty_match_on_empty_enum_to_anything(x: UninhabitedEnum) -> A {
    match x {}
}

fn cannot_empty_match_on_empty_struct_to_anything(x: UninhabitedStruct) -> A {
    match x {}
}

fn cannot_empty_match_on_empty_tuple_struct_to_anything(x: UninhabitedTupleStruct) -> A {
    match x {}
}

fn cannot_empty_match_on_enum_with_empty_variants_struct_to_anything(x: UninhabitedVariants) -> A {
    match x {}
}

fn main() {}
