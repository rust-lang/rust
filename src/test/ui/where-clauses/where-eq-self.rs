#![feature(type_equality_constraints)]

pub struct Foo;

impl Foo {
    pub fn foo<T>(&self) where T = Self {}
}

pub trait Bar {
    fn test<T>() -> Self where T = Self;
}

impl Bar for Foo {
    fn test<T>() -> Self where T = Self {
    //~^ ERROR method not compatible with trait
      Foo
    }
}

fn main() {}
