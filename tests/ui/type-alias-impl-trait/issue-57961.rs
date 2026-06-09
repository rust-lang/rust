#![feature(type_alias_impl_trait)]

type X = impl Sized;

trait Foo {
    type Bar: Iterator<Item = X>;
}

impl Foo for () {
    type Bar = std::vec::IntoIter<u32>;
    //~^ ERROR expected `IntoIter<u32>` to be an iterator that yields `X`, but it yields `u32`
}

#[define_opaque(X)]
fn incoherent() -> X {
    22_i32
}

fn main() {}
