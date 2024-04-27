//@ run-pass
#![feature(associated_const_equality)]
#![allow(unused)]

pub trait Foo {
  const N: usize;
}

pub struct Bar;

impl Foo for Bar {
  const N: usize = 3;
}

const TEST:usize = 3;


fn foo<F: Foo<N=3usize>>() {}

fn main() {
  foo::<Bar>()
}
