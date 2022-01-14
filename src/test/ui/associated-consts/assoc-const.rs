#![feature(associated_const_equality)]

pub trait Foo {
  const N: usize;
}

pub struct Bar;

impl Foo for Bar {
  const N: usize = 3;
}

const TEST:usize = 3;


fn foo<F: Foo<N=3>>() {}
//~^ ERROR associated const equality is incomplete
fn bar<F: Foo<N={TEST}>>() {}
//~^ ERROR associated const equality is incomplete

fn main() {}
