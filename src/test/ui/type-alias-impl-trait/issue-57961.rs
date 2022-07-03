#![feature(type_alias_impl_trait)]

type X = impl Sized;

trait Foo {
    type Bar: Iterator<Item = X>;
}

impl Foo for () {
    type Bar = std::vec::IntoIter<u32>;
    //~^ ERROR expected `std::vec::IntoIter<u32>` to be an iterator of `X`, but it actually returns items of `u32`
}

fn incoherent() {
    let f: X = 22_i32;
}

fn main() {}
