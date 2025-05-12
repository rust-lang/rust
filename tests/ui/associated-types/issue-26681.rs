#![feature(associated_type_defaults)]

// This is a partial regression test for #26681, which used to fail to resolve
// `Self` in the assoc. constant, and now fails with a type mismatch because
// `Self::Fv` cannot be assumed to equal `u8` inside the trait.

trait Foo {
    type Bar;
}

impl Foo for u8 {
    type Bar = ();
}

trait Baz {
    type Fv: Foo = u8;
    const C: <Self::Fv as Foo>::Bar = 6665;  //~ error: mismatched types
}

fn main() {}
