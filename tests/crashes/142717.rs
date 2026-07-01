//@ known-bug: rust-lang/rust#142717
#![feature(inherent_associated_types)]
struct Foo<T>(T);

impl<'a> Foo<fn(&'a ())> {
    type Assoc = &mut Peekable<I>;
}

fn bar(_: for<'a> fn(Foo<fn(Foo<fn(&'a ())>::Assoc)>::Assoc)) {}

pub fn main() {}
