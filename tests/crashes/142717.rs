//@ known-bug: #142717
//@ compile-flags: -Znext-solver=coherence

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo<T>(T);

impl<'a> Foo<fn(&'a ())> {
    type Assoc = &mut Peekable<I>;
}

fn bar(_: for<'a> fn(Foo<fn(Foo<fn(&'a ())>::Assoc)>::Assoc)) {}

fn main() {}
