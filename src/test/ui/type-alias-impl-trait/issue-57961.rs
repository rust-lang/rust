#![feature(type_alias_impl_trait)]

type X = impl Sized;

trait Foo {
    type Bar: Iterator<Item = X>;
}

impl Foo for () {
    type Bar = std::vec::IntoIter<u32>;
    //~^ ERROR type mismatch resolving `<std::vec::IntoIter<u32> as Iterator>::Item == X
}

fn incoherent() {
    let f: X = 22_i32;
}

fn main() {}
