#![feature(associated_const_equality)]
#![allow(unused)]

pub trait Foo {
    const N: usize;
}

pub trait FooTy {
    type T;
}

pub struct Bar;

impl Foo for Bar {
    const N: usize = 3;
}

impl FooTy for Bar {
    type T = usize;
}


fn foo<F: Foo<N = usize>>() {}
//~^ ERROR expected constant, found type
fn foo2<F: FooTy<T = 3usize>>() {}
//~^ ERROR expected type, found constant

fn main() {
    foo::<Bar>();
    foo2::<Bar>();
}
