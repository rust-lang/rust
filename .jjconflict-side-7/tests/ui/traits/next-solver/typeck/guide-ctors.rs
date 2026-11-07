//@ compile-flags: -Znext-solver
//@ check-pass

// Makes sure we structurally normalize before trying to use expectation to guide
// coercion in adt and tuples.

use std::any::Any;

trait Coerce {
    type Assoc;
}

struct TupleGuidance;
impl Coerce for TupleGuidance {
    type Assoc = (&'static dyn Any,);
}

struct AdtGuidance;
impl Coerce for AdtGuidance {
    type Assoc = Adt<&'static dyn Any>;
}

struct Adt<T> {
    f: T,
}

fn foo<'a, T: Coerce>(_: T::Assoc) {}

fn main() {
    foo::<TupleGuidance>((&0u32,));
    foo::<AdtGuidance>(Adt { f: &0u32 });
}
