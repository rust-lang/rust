#![feature(fn_traits, unboxed_closures)]

struct Foo;

impl<'a, T> Fn<(&'a T,)> for Foo {
  extern "rust-call" fn call(&self, (_,): (T,)) {}
  //~^ ERROR: has an incompatible type for trait
}

impl<'a, T> FnMut<(&'a T,)> for Foo {
  extern "rust-call" fn call_mut(&mut self, (_,): (T,)) {}
  //~^ ERROR: has an incompatible type for trait
}

impl<'a, T> FnOnce<(&'a T,)> for Foo {
  type Output = ();

  extern "rust-call" fn call_once(self, (_,): (T,)) {}
  //~^ ERROR: has an incompatible type for trait
}

fn main() {}
