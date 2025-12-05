#![feature(never_type)]

#[non_exhaustive]
pub enum UninhabitedEnum {
}

#[non_exhaustive]
pub struct UninhabitedTupleStruct(!);

#[non_exhaustive]
pub struct UninhabitedStruct {
    _priv: !,
}

pub enum UninhabitedVariants {
    #[non_exhaustive] Tuple(!),
    #[non_exhaustive] Struct { x: ! }
}

struct A;

// This test checks that uninhabited non-exhaustive types defined in the same crate cannot coerce
// to any type, as the never type can.

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
