#![feature(associated_const_equality)]
#![allow(unused)]

pub trait Foo {
  const N: usize;
}

pub struct Bar;

impl Foo for Bar {
  const N: usize = 3;
}

fn foo1<F: Foo<Z = 3>>() {}
//~^ ERROR associated constant `Z` not found for `Foo`
fn foo2<F: Foo<Z = usize>>() {}
//~^ ERROR associated type `Z` not found for `Foo`
fn foo3<F: Foo<Z = 5>>() {}
//~^ ERROR associated constant `Z` not found for `Foo`

fn main() {
  foo1::<Bar>();
  foo2::<Bar>();
  foo3::<Bar>();
}
