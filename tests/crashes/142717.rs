//@ known-bug: #142717
//@ compile-flags: -Znext-solver=coherence

#![feature(inherent_associated_types)]

struct Type<T>(T);

impl<'a> Type<fn(&'a ())> {
    type Assoc = ();
}

fn bar(_: for<'a> fn(Type<fn(Type<fn(&'a ())>::Assoc)>::Assoc)) {}

fn main() {}
