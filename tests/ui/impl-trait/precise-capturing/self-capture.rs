//@ check-pass

#![feature(precise_capturing_in_traits)]

trait Foo {
    fn bar<'a>() -> impl Sized + use<Self>;
}

fn main() {}
