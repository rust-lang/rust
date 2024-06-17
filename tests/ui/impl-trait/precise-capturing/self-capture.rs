//@ check-pass

#![feature(precise_capturing)]

trait Foo {
    fn bar<'a>() -> impl Sized + use<Self>;
}

fn main() {}
