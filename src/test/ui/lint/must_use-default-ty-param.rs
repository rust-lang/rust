//! Tests that the lowering of `#[must_use]` compiles in presence of defaulted type parameters.

// check-pass

#![deny(unused_must_use)]

trait Tr {
    type Assoc;
}

#[must_use]
struct Generic<T: Tr, U = <T as Tr>::Assoc>(T, U);

fn main() {}
