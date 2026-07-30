//@ compile-flags: -Znext-solver

// Regression test for https://github.com/rust-lang/rust-clippy/issues/17439

#![feature(inherent_associated_types)]

struct Foo<T>(T);

impl<'a> Foo<fn(&'a ())> {
    type Assoc = &'a ();
}

fn bar(_: fn(Foo<for<'b> fn(Foo<fn(&'b ())>::Assoc)>::Assoc)) {}
//~^ ERROR: higher-ranked subtype error
//~| ERROR: higher-ranked subtype error
//~| ERROR: lifetime bound not satisfied [E0478]
//~| ERROR: lifetime bound not satisfied [E0478]

fn main() {}
