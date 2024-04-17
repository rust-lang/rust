//@ check-pass

#![feature(precise_capturing)]
//~^ WARN the feature `precise_capturing` is incomplete

trait Foo {
    fn bar<'a>() -> impl use<Self> Sized;
}

fn main() {}
