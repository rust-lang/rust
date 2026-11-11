//@ check-pass

#![feature(rustc_attrs)]

trait Foo {
    type Bar;
}

impl Foo for () {
    type Bar = u32;
}

fn foo() -> <() as Foo>::Bar {
    22
}

fn main() { }
