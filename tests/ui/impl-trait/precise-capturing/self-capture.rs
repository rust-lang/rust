//@ check-pass

#![feature(precise_capturing)]
//~^ WARN the feature `precise_capturing` is incomplete

trait Foo {
    fn bar<'a>() -> impl Sized + use<Self>;
}

fn main() {}
