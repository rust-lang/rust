// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="rlib"]

pub trait Foo {
  fn generic_method<T>();
}

pub struct Bar;

impl Foo for Bar {
  fn generic_method<T>() {}
}

pub fn mono_function() {
  Bar::generic_method::<Bar>();
}

pub fn mono_function_lifetime<'a>(x: &'a u64) -> u64 {
  *x
}

pub fn generic_function<T>(t: T) -> T {
  t
}

pub fn user() {
  generic_function(0u32);
  generic_function("abc");
  let x = 2u64;
  generic_function(&x);
  let _ = mono_function_lifetime(&x);
}
