#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

trait Bar<const N: usize> {}

trait Foo<'a> {
    const N: usize;
    type Baz: Bar<{ Self::N }>;
    //~^ ERROR: unconstrained generic constant
}

fn main() {}
