// run-pass
#![feature(const_generics)]
#![allow(incomplete_features)]

trait Foo {
    const ASSOC: usize;
}

impl Foo for &'static () {
    const ASSOC: usize = 13;
}

fn test<'a>() where &'a (): Foo {
    let _: [u8; <&'a () as Foo>::ASSOC];
}

fn main() {
    test();
}
