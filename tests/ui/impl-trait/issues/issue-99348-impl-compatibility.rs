#![feature(type_alias_impl_trait)]

struct Concrete;

type Tait = impl Sized;

impl Foo for Concrete {
    type Item = Concrete;
    //~^ type mismatch resolving
}

impl Bar for Concrete {
    type Other = Tait;
}

trait Foo {
    type Item: Bar<Other = Self>;
}

trait Bar {
    type Other;
}

fn tait() -> Tait {}

fn main() {}
