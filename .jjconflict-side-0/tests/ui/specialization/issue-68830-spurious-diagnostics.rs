// A regression test for #68830. This checks we don't emit
// a verbose `conflicting implementations` error.

#![feature(specialization)]
#![allow(incomplete_features)]

struct BadStruct {
    err: MissingType //~ ERROR: cannot find type `MissingType` in this scope
}

trait MyTrait<T> {
    fn foo();
}

impl<T, D> MyTrait<T> for D {
    default fn foo() {}
}

impl<T> MyTrait<T> for BadStruct {
    fn foo() {}
}

fn main() {}
