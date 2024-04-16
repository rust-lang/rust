//@ known-bug: #122710
use std::marker::PhantomData;

pub trait BarTrait<T> {
  fn bar(self, _: T);
}

impl<T, F: Fn(T)> BarTrait<T> for F {
  fn bar(self, _: T) { }
}

impl<T: for<'a> MyTrait<'a>> BarTrait<T> for () {
  fn bar(self, _: T) { }
}

pub trait MyTrait<'a> { }

impl<'a> MyTrait<'a> for PhantomData<&'a ()> { }

fn foo() {
  ().bar(PhantomData);
}

pub fn main() {}
