//@ run-pass
#![feature(min_generic_const_args)]
#![allow(unused, incomplete_features)]

pub trait Foo {
  #[type_const]
  const N: usize;
}

pub struct Bar;

impl Foo for Bar {
  #[type_const]
  const N: usize = 3;
}

const TEST: usize = 3;


fn foo<F: Foo<N = 3usize>>() {}

fn main() {
  foo::<Bar>()
}
