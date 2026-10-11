// Regression test for issue #163792: ensure that type mismatch diagnostics between
// the same ADT with different generic arguments do not omit differing default parameters
// in the inline label (e.g. expected `Foo<_, ()>`, found `Foo<(), u32>`).

use std::marker::PhantomData;

struct Foo<A, B = ()> {
    _phantom: PhantomData<(A, B)>,
}

struct Bar<B = ()> {
    _phantom: PhantomData<B>,
}

impl<B> Bar<B> {
    fn call_bar<A>(&self, foo: Foo<A, B>) -> Foo<A, B> {
        foo
    }
}

fn abc(bar: Bar, foo: Foo<(), u32>) -> Foo<(), u32> {
    bar.call_bar(foo)
    //~^ ERROR mismatched types
}

fn main() {}
