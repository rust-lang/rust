// Test for issue #116766.
// Ensure that we don't suggest impl'ing `Copy` for a type if it or at least one
// of it's variants are marked as `non_exhaustive`.

//@ check-pass

#![deny(missing_copy_implementations)]

#[non_exhaustive]
pub enum MyEnum {
    A,
}

#[non_exhaustive]
pub struct MyStruct {
    foo: usize,
}

pub enum MyEnum2 {
    #[non_exhaustive]
    A,
    B,
}

fn main() {}
