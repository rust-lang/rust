// Test that AsRepr is not implemented for enums with non_exhaustive attributes.

// gate-test-enum_as_repr

#![feature(enum_as_repr)]

use std::enums::AsRepr;

#[repr(u8)]
#[non_exhaustive]
enum AttrOnEnum {
    Zero,
    One,
}

#[repr(u8)]
enum AttrOnVariant {
    Zero,
    #[non_exhaustive]
    One,
}

fn main() {
    consumes_asrepr(AttrOnEnum::Zero);
    //~^ ERROR the trait bound `AttrOnEnum: HasAsReprImpl` is not satisfied [E0277]

    consumes_asrepr(AttrOnVariant::Zero);
    //~^ ERROR the trait bound `AttrOnVariant: HasAsReprImpl` is not satisfied [E0277]
}

fn consumes_asrepr<V: AsRepr>(_v: V) {}
