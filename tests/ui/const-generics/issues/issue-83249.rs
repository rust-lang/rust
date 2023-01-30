#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

trait Foo {
    const N: usize;
}

impl Foo for u8 {
    const N: usize = 1;
}

fn foo<T: Foo>(_: [u8; T::N]) -> T {
    todo!()
}

pub fn bar() {
    let _: u8 = foo([0; 1]);

    let _ = foo([0; 1]);
    //~^ ERROR type annotations needed
}

fn main() {}
