//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo<T>(T);

impl<'a> Foo<fn(&'a ())> {
    type Assoc = &'a ();
}

fn bar(_: fn(Foo<for<'b> fn(Foo<fn(&'b ())>::Assoc)>::Assoc)) {}
//[current]~^ ERROR mismatched types [E0308]
//[current]~| ERROR mismatched types [E0308]
//[current]~| ERROR higher-ranked subtype error
//[current]~| ERROR higher-ranked subtype error
//[current]~| ERROR higher-ranked subtype error
//[next]~^^^^^^ ERROR type mismatch resolving
//[next]~| ERROR type mismatch resolving

fn main() {}
