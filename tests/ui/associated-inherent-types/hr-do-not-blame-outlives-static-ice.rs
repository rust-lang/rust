//@ compile-flags: -Zdeduplicate-diagnostics=yes

// Regression test for #146467.
#![feature(inherent_associated_types)]
//~^ WARN the feature `inherent_associated_types` is incomplete

struct Foo<T>(T);

impl<'a> Foo<fn(&())> {
    //~^ ERROR the lifetime parameter `'a` is not constrained by the impl trait
    type Assoc = &'a ();
}

fn foo(_: for<'a> fn(Foo<fn(&'a ())>::Assoc)) {}
//~^ ERROR mismatched types
//~| ERROR higher-ranked subtype error
fn main() {}
