// Regression test for an NLL-related ICE (#52992) -- computing
// implied bounds was causing outlives relations that were not
// properly handled.
//
//@ check-pass

fn main() {}

fn fail<'a>() -> Struct<'a, Generic<()>> {
    Struct(&Generic(()))
}

struct Struct<'a, T>(&'a T) where
    T: Trait + 'a,
    T::AT: 'a; // only fails with this bound

struct Generic<T>(T);

trait Trait {
    type AT;
}

impl<T> Trait for Generic<T> {
    type AT = T; // only fails with a generic AT
}
