#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// Check that we don't crash when printing inherent projections in diagnostics.

struct Foo<T>(T);

impl<'a> Foo<fn(&'a ())> {
    type Assoc = &'a ();
}

fn main(_: for<'a> fn(Foo<fn(&'a ())>::Assoc)) {} //~ ERROR `main` function has wrong type
