//@ check-pass
#![deny(unreachable_patterns)]
#![feature(never_type)]

#[non_exhaustive]
pub enum UninhabitedEnum {
}

#[non_exhaustive]
pub struct UninhabitedStruct {
    pub never: !,
    _priv: (),
}

#[non_exhaustive]
pub struct UninhabitedTupleStruct(pub !);

pub enum UninhabitedVariants {
    #[non_exhaustive] Tuple(!),
    #[non_exhaustive] Struct { x: ! }
}

struct A;

// This checks that `non_exhaustive` annotations do not affect exhaustiveness checking within the
// defining crate.
fn empty_match_on_empty_enum(x: UninhabitedEnum) -> A {
    match x {}
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
