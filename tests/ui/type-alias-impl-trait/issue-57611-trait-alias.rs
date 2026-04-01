//@ check-pass
// Regression test for issue #57611
// Ensures that we don't ICE

#![feature(trait_alias)]
#![feature(impl_trait_in_assoc_type)]

trait Foo {
    type Bar: Baz<Self, Self>;

    fn bar(&self) -> Self::Bar;
}

struct X;

impl Foo for X {
    type Bar = impl Baz<Self, Self>;

    fn bar(&self) -> Self::Bar {
        |x| x
    }
}

trait Baz<A: ?Sized, B: ?Sized> = Fn(&A) -> &B;

fn main() {}
