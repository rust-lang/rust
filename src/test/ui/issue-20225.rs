// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(fn_traits, unboxed_closures)]

struct Foo;

impl<'a, T> Fn<(&'a T,)> for Foo {
  extern "rust-call" fn call(&self, (_,): (T,)) {}
  //~^ ERROR: has an incompatible type for trait
  //~| expected reference
}

impl<'a, T> FnMut<(&'a T,)> for Foo {
  extern "rust-call" fn call_mut(&mut self, (_,): (T,)) {}
  //~^ ERROR: has an incompatible type for trait
  //~| expected reference
}

impl<'a, T> FnOnce<(&'a T,)> for Foo {
  type Output = ();

  extern "rust-call" fn call_once(self, (_,): (T,)) {}
  //~^ ERROR: has an incompatible type for trait
  //~| expected reference
}

fn main() {}
