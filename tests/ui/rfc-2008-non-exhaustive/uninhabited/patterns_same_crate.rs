#![deny(unreachable_patterns)]
#![feature(exhaustive_patterns)]
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

pub enum PartiallyInhabitedVariants {
    Tuple(u8),
    #[non_exhaustive] Struct { x: ! }
}

fn uninhabited_enum() -> Option<UninhabitedEnum> {
    None
}

fn uninhabited_variant() -> Option<UninhabitedVariants> {
    None
}

fn partially_inhabited_variant() -> PartiallyInhabitedVariants {
    PartiallyInhabitedVariants::Tuple(3)
}

fn uninhabited_struct() -> Option<UninhabitedStruct> {
    None
}

fn uninhabited_tuple_struct() -> Option<UninhabitedTupleStruct> {
    None
}

// This test checks that non-exhaustive types that would normally be considered uninhabited within
// the defining crate are still considered uninhabited.

fn main() {
    match uninhabited_enum() {
        Some(_x) => (), //~ ERROR unreachable pattern
        None => (),
    }

    match uninhabited_variant() {
        Some(_x) => (), //~ ERROR unreachable pattern
        None => (),
    }

    while let PartiallyInhabitedVariants::Struct { x } = partially_inhabited_variant() {
        //~^ ERROR unreachable pattern
    }

    while let Some(_x) = uninhabited_struct() { //~ ERROR unreachable pattern
    }

    while let Some(_x) = uninhabited_tuple_struct() { //~ ERROR unreachable pattern
    }
}
