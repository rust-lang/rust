#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo<T>(T);

impl<'a> Foo<fn(&'a ())> {
    type Assoc = &'a ();
}

fn bar(_: fn(Foo<for<'b> fn(Foo<fn(&'b ())>::Assoc)>::Assoc)) {}
//~^ ERROR mismatched types [E0308]
//~| ERROR mismatched types [E0308]
//~| ERROR higher-ranked subtype error
//~| ERROR higher-ranked subtype error

fn main() {}
