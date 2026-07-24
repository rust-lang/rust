//! Regression test for <https://github.com/rust-lang/rust/issues/104637>.
//! A failed bound on an explicitly typed struct should not cause duplicate type errors.

trait Trait {}

struct Struct<T>
where
    Struct<T>: Trait,
{
    field: T,
}

impl Trait for Struct<bool> {}

fn main() {
    let _: Struct<u8> = Struct { field: 0 };
    //~^ ERROR the trait bound `Struct<u8>: Trait` is not satisfied
    //~| ERROR mismatched types
    //~| ERROR mismatched types
}
