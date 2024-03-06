// This test verifies that we do not ICE due to MIR inlining in case of normalization failure
// in a projection.
//
//@ compile-flags: --crate-type lib -C opt-level=3
//@ build-pass

pub trait Trait {
    type Associated;
}
impl<T> Trait for T {
    type Associated = T;
}

pub struct Struct<T>(<T as Trait>::Associated);

pub fn foo<T>() -> Struct<T>
where
    T: Trait,
{
    bar()
}

#[inline]
fn bar<T>() -> Struct<T> {
    Struct(baz())
}

fn baz<T>() -> T {
    unimplemented!()
}
