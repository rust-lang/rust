// Test that the variance computation considers types/regions that
// appear in projections to be invariant.

#![feature(rustc_attrs)]

trait Trait<'a> {
    type Type;

    fn method(&'a self) { }
}

#[rustc_variance]
struct Foo<'a, T : Trait<'a>> { //~ ERROR ['a: +, T: +]
    field: (T, &'a ())
}

#[rustc_variance]
struct Bar<'a, T : Trait<'a>> { //~ ERROR ['a: o, T: o]
    field: <T as Trait<'a>>::Type
}

fn main() { }
