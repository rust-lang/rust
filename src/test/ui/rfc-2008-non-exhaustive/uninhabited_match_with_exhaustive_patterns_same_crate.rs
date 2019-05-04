// compile-pass
#![deny(unreachable_patterns)]
#![feature(exhaustive_patterns)]
#![feature(never_type)]
#![feature(non_exhaustive)]

#[non_exhaustive]
pub enum UninhabitedEnum {
}

struct A;

// This test checks that an empty match on a non-exhaustive uninhabited type from the defining crate
// will compile. In particular, this enables the `exhaustive_patterns` feature as this can
// change the branch used in the compiler to determine this.

fn cannot_empty_match_on_empty_enum_to_anything(x: UninhabitedEnum) -> A {
    match x {}
}

fn main() {}
